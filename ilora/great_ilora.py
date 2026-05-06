from great.base.great_base import BaseGReaT
import logging
import typing as tp
import torch
import torch.nn as nn

# from great.base.great_base import BaseGReaT

# internal relative imports within ilora/
from .ilora_linear import ILoRALinear
from .importance_scorer import ImportanceScorer
from .utils import replace_with_ilora, print_ilora_parameters


class GReaTILoRA(BaseGReaT):
    """
    I-LoRA: Importance-based Low-Rank Adaptation for GReaT.
    
    Unlike LoRA/DoRA which update ALL neurons with low-rank matrices,
    I-LoRA first identifies the most task-relevant neurons via a 
    calibration pass, then applies low-rank updates ONLY to those.
    
    Math:  W' = W_0 + Pi_S @ B_S @ A
    """

    def __init__(
        self,
        llm,
        method,
        experiment_dir=None,
        epochs=1,
        batch_size=1,
        report_to=None,
        train_kwargs=None,
        ilora_cfg=None,          # I-LoRA specific config
        float_precision=None,
        calibration_dataloader=None,   # needed for importance scoring
    ):
        super().__init__(
            llm=llm,
            method=method,
            experiment_dir=experiment_dir,
            epochs=epochs,
            batch_size=batch_size,
            report_to=report_to,
            train_kwargs=train_kwargs,
            float_precision=float_precision,
        )
        self.ilora_cfg = ilora_cfg or {}
        self.calibration_dataloader = calibration_dataloader

    @staticmethod
    def _detect_target_modules(model) -> tp.List[str]:
        """Same auto-detection logic as LoRA/DoRA."""
        candidate_patterns = [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["c_attn", "c_proj"],
            ["c_attn"],
            ["query_key_value", "dense"],
            ["query_key_value"],
            ["q_proj", "k_proj", "v_proj", "out_proj"],
            ["q_proj", "v_proj", "k_proj", "out_proj"],
        ]
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        for pattern in candidate_patterns:
            if all(p in module_names for p in pattern):
                logging.info(f"I-LoRA target modules: {pattern}")
                return pattern

        fallback = set()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                short = name.split(".")[-1]
                if any(kw in short.lower() for kw in
                       ("attn", "attention", "proj", "query", "key", "value")):
                    fallback.add(short)
        if fallback:
            result = sorted(fallback)
            logging.info(f"I-LoRA target modules (fallback): {result}")
            return result

        raise ValueError("Could not auto-detect target modules for I-LoRA.")

    #     # ── Step 3: Report ────────────────────────────────────────────────
    #     print("\nI-LoRA model ready:")
    #     print_ilora_parameters(self.model)

    def _apply_model_modifications(self):

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print()
                print(f"Linear layer: {name} | shape: {module.weight.shape}")
                print()
                cfg = self.ilora_cfg
        for name, module in self.model.named_modules():
            if isinstance(module, ILoRALinear):
                total = sum(p.numel() for p in module.parameters() if p.requires_grad)
                frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
                print(f"{name}: trainable={total:,} frozen={frozen:,} | B_S={module.B_S.shape} A={module.A.shape}")

        r             = cfg.get("r", 8)
        lora_alpha    = cfg.get("lora_alpha", 16.0)
        dropout       = cfg.get("dropout", 0.05)
        top_k_ratio   = cfg.get("top_k_ratio", 0.3)
        n_calib_batch = cfg.get("n_calibration_batches", 50)

        target_modules = cfg.get("target_modules", None)
        if target_modules is None:
            target_modules = self._detect_target_modules(self.model)

        print(f"Target modules to adapt: {target_modules}")

        # ── Step 1: Compute importance ────────────────────────────────────
        print("Computing neuron importance scores (calibration pass)...")
        scorer = ImportanceScorer(
            model=self.model,
            tokenizer=self.tokenizer,
            n_batches=n_calib_batch,
        )
        importance_scores = scorer.compute(self.calibration_dataloader)
        importance_masks  = scorer.get_masks(top_k_ratio=top_k_ratio)

        # ── DEBUG — remove after confirming it works ──────────────────────
        print(f"Total layers scored: {len(importance_masks)}")
        print(f"Sample keys: {list(importance_masks.keys())[:5]}")

        # ── Step 2: Filter to target modules ─────────────────────────────
        # importance_masks keys are FULL names: "transformer.h.0.attn.c_attn"
        # target_modules are SHORT names: "c_attn"
        # so we match using the SHORT name (last part after final dot)
        filtered_masks = {
            full_name: indices
            for full_name, indices in importance_masks.items()
            if full_name.split(".")[-1] in target_modules
        }

        print(f"Found important neurons in {len(filtered_masks)} layers")
        if len(filtered_masks) == 0:
            raise ValueError(
                f"No layers matched target_modules={target_modules}. "
                f"Available short names: "
                f"{set(k.split('.')[-1] for k in importance_masks.keys())}"
            )

        # ── Step 3: Replace layers ────────────────────────────────────────
        self.model = replace_with_ilora(
            model=self.model,
            importance_masks=filtered_masks,   # full names as keys
            r=r,
            lora_alpha=lora_alpha,
            dropout=dropout,
            target_module_names=target_modules,  # short names for matching
        )
        # ── Step 3.2: Explicitly freeze everything, then unfreeze only I-LoRA params ──
        print("Freezing base weights, keeping only I-LoRA params trainable...")

        # First freeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Then unfreeze ONLY B_S and A in ILoRALinear modules
        for name, module in self.model.named_modules():
            if isinstance(module, ILoRALinear):
                module.B_S.requires_grad_(True)
                module.A.requires_grad_(True)

        # ── Step 4: Report ────────────────────────────────────────────────
        print("\nI-LoRA model ready:")
        print_ilora_parameters(self.model)
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D   # ← ADD THIS IMPORT
from .ilora_linear import ILoRALinear


def replace_with_ilora(
    model,
    importance_masks,
    r=8,
    lora_alpha=16.0,
    dropout=0.05,
    target_module_names=None,
):
    def _get_parent(model, full_name):
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    replaced = 0

    for full_name, important_indices in importance_masks.items():
        # Navigate to module
        try:
            module = model
            for part in full_name.split('.'):
                module = getattr(module, part)
        except AttributeError:
            print(f"Warning: could not find module {full_name}, skipping")
            continue

        # ← Accept both nn.Linear and Conv1D
        if not isinstance(module, (nn.Linear, Conv1D)):
            print(f"Warning: {full_name} is {type(module)}, skipping")
            continue

        ilora_layer = ILoRALinear(
            original_linear=module,
            important_indices=important_indices,
            r=r,
            lora_alpha=lora_alpha,
            dropout=dropout,
        )

        # Fix the argument — was a bug in previous version
        ilora_layer = ILoRALinear(
            original_linear=module,
            important_indices=important_indices,
            r=r,
            lora_alpha=lora_alpha,
            dropout=dropout,
        )

        parent, attr = _get_parent(model, full_name)
        setattr(parent, attr, ilora_layer)
        replaced += 1
        print(f"  Replaced: {full_name} ({len(important_indices)}/{module.weight.shape[0]} neurons)")

    print(f"\nTotal replaced: {replaced} layers")
    return model


# def print_ilora_parameters(model):
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total     = sum(p.numel() for p in model.parameters())
#     print(f"Trainable parameters: {trainable:,}")
#     print(f"Total parameters:     {total:,}")
#     print(f"Trainable ratio:      {100 * trainable / total:.4f}%")

def print_ilora_parameters(model):
    from .ilora_linear import ILoRALinear

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable ratio:      {100 * trainable / total:.4f}%")

    # print("\nPer-layer matrix sizes:")
    # print(f"  {'Layer':<45} {'m':>6} {'n':>6} {'k':>6} {'r':>4}  |  Ps(m×k)       B_S(k×r)    A(r×n)")
    # print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*4}  |  {'-'*12}  {'-'*10}  {'-'*10}")

    # for name, module in model.named_modules():
    #     if isinstance(module, ILoRALinear):
    #         m = module.n_out
    #         n = module.n_in
    #         k = module.k
    #         r = module.r
    #         print(
    #             f"  {name:<45} {m:>6} {n:>6} {k:>6} {r:>4}"
    #             f"  |  {m}×{k:<8}  {k}×{r:<8}  {r}×{n}"
    #         )
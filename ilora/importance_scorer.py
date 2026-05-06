import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D  # ← ADD THIS IMPORT
from .ilora_linear import ILoRALinear


class ImportanceScorer:
    def __init__(self, model, tokenizer, n_batches=50):
        self.model = model
        self.tokenizer = tokenizer
        self.n_batches = n_batches
        self.importance_scores = {}

    def compute(self, dataloader) -> dict:

        # Temporarily enable gradients for ALL parameters
        original_grad_state = {}
        for name, param in self.model.named_parameters():
            original_grad_state[name] = param.requires_grad
            param.requires_grad_(True)

        self.model.train()

        # ── Initialize for ALL Linear AND Conv1D layers ───────────────
        # for name, module in self.model.named_modules():
        #     if isinstance(module, (nn.Linear, Conv1D)):   # ← Conv1D added
        #         self.importance_scores[name] = torch.zeros(
        #             module.weight.shape[0]
        #         )

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                if isinstance(module, Conv1D):
                    n_out = module.weight.shape[1]   # (n_in, n_out)
                else:
                    n_out = module.weight.shape[0]   # (n_out, n_in)

                self.importance_scores[name] = torch.zeros(n_out)

        print(f"Scoring {len(self.importance_scores)} Linear/Conv1D layers...")

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.n_batches:
                break

            batch = {
                k: v.to(next(self.model.parameters()).device)
                for k, v in batch.items()
            }

            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            # for name, module in self.model.named_modules():
            #     if isinstance(module, (nn.Linear, Conv1D)):   # ← Conv1D added
            #         if module.weight.grad is not None:
            #             grad  = module.weight.grad.detach()
            #             weight = module.weight.data
            #             score = (grad * weight).norm(dim=1).cpu()
            #             self.importance_scores[name] += score

            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, Conv1D)):
                    if module.weight.grad is not None:
                        grad   = module.weight.grad.detach()
                        weight = module.weight.data

                        if isinstance(module, Conv1D):
                            # Conv1D weight: (n_in, n_out)
                            # output dim = dim 1 → norm along dim 0 (input dim)
                            score = (grad * weight).norm(dim=0).cpu()  # shape: (n_out,)
                        else:
                            # nn.Linear weight: (n_out, n_in)
                            # output dim = dim 0 → norm along dim 1 (input dim)
                            score = (grad * weight).norm(dim=1).cpu()  # shape: (n_out,)

                        self.importance_scores[name] += score

            print(f"  Batch {batch_idx+1}/{self.n_batches} done", end="\r")

        print()

        # Restore original grad state
        for name, param in self.model.named_parameters():
            param.requires_grad_(original_grad_state[name])

        # Remove layers with all-zero scores
        self.importance_scores = {
            name: scores
            for name, scores in self.importance_scores.items()
            if scores.sum() > 0
        }

        print(f"Layers with non-zero importance: {len(self.importance_scores)}")

        # Normalize per layer
        for name in self.importance_scores:
            s = self.importance_scores[name]
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                self.importance_scores[name] = (s - s_min) / (s_max - s_min)
            else:
                self.importance_scores[name] = torch.ones_like(s)

        return self.importance_scores

    def get_masks(self, top_k_ratio=0.3) -> dict:
        masks = {}
        for name, scores in self.importance_scores.items():
            k = max(1, int(top_k_ratio * len(scores)))
            _, indices = scores.topk(k)
            masks[name] = indices.sort().values
        return masks
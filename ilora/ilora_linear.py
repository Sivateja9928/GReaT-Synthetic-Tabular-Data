import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class ILoRALinear(nn.Module):
    def __init__(
        self,
        original_linear,
        important_indices,
        r=8,
        lora_alpha=16.0,
        dropout=0.05,
    ):
        super().__init__()

        # ── Extract weight correctly for both Conv1D and nn.Linear ────
        if isinstance(original_linear, Conv1D):
            # Conv1D weight: (n_in, n_out)
            # Conv1D forward does: x @ weight + bias
            # which means output dim = n_out = weight.shape[1]
            # We store it as-is and use matmul in forward
            weight_data = original_linear.weight.data.clone()  # (n_in, n_out)
            bias_data   = original_linear.bias.data.clone() if original_linear.bias is not None else None
            self.is_conv1d = True
            n_in, n_out = weight_data.shape   # (in, out)
        else:
            # nn.Linear weight: (n_out, n_in)
            # nn.Linear forward does: x @ weight.T + bias
            weight_data = original_linear.weight.data.clone()  # (n_out, n_in)
            bias_data   = original_linear.bias.data.clone() if original_linear.bias is not None else None
            self.is_conv1d = False
            n_out, n_in = weight_data.shape   # (out, in)

        # ── Frozen pretrained weight ──────────────────────────────────
        self.weight = nn.Parameter(weight_data, requires_grad=False)

        if bias_data is not None:
            self.bias = nn.Parameter(bias_data, requires_grad=False)
        else:
            self.bias = None

        # ── Important neuron indices (into output dimension) ──────────
        self.register_buffer('important_indices', important_indices)

        k = len(important_indices)

        # ── Trainable I-LoRA parameters ───────────────────────────────
        # A: projects input (n_in) → rank r
        self.A   = nn.Parameter(torch.empty(r, n_in))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

        # B_S: projects rank r → k important output neurons
        self.B_S = nn.Parameter(torch.zeros(k, r))

        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.n_in  = n_in
        self.n_out = n_out
        self.k     = k
        self.r     = r

    def forward(self, x):
        # ── Base frozen forward ───────────────────────────────────────
        if self.is_conv1d:
            # Conv1D: output = x @ W + b   (W is n_in x n_out)
            base_out = x @ self.weight
            if self.bias is not None:
                base_out = base_out + self.bias
        else:
            # nn.Linear: output = x @ W.T + b   (W is n_out x n_in)
            base_out = F.linear(x, self.weight, self.bias)

        # ── I-LoRA delta ──────────────────────────────────────────────
        x_dropped = self.dropout(x)             # (..., n_in)
        lora_out  = x_dropped @ self.A.T        # (..., r)
        lora_out  = lora_out  @ self.B_S.T      # (..., k)

        # Scatter k values into n_out positions
        delta = torch.zeros(
            *x.shape[:-1], self.n_out,
            device=x.device,
            dtype=x.dtype,
        )
        delta[..., self.important_indices] = lora_out   # (..., n_out)

        return base_out + self.scaling * delta

    def get_trainable_param_count(self):
        return self.B_S.numel() + self.A.numel()
"""
Topographic channel pooling after Qian et al. (2026, Nat Commun 17:4042).

The C channels of an area are laid out on a sqrt(C) x sqrt(C) torus ("sheet"). After the
nonlinearity, every channel's response is replaced by a Gaussian-weighted average over its
sheet neighbours at the same pixel:

    y[c] = sum_c' K[c, c'] x[c'],   K[c, c'] ∝ exp(-d(c, c')² / 2σ²),   rows of K sum to 1

Neighbouring channels are forced to share output, so training makes neighbouring kernels
similar and feature maps form on the sheet. σ starts at half the sheet and shrinks with the
epoch (see DyRCNN._update_sheetpool_sigma), down to a floor.
"""

import torch
import torch.nn as nn

__all__ = ["SheetPool"]


class SheetPool(nn.Module):
    def __init__(self, n_channels: int, sigma_floor: float = 0.75, mix: float = 1.0) -> None:
        super().__init__()
        self.mix = float(mix) # y = (1 - mix)·x + mix·Kx; 1 = pure Qian pooling
        side = int(round(n_channels**0.5))
        if side * side != n_channels:
            raise ValueError(f"SheetPool needs a square channel count, got {n_channels}")
        self.side = side
        self.sigma_floor = float(sigma_floor)

        idx = torch.arange(side)
        pos = torch.stack(torch.meshgrid(idx, idx, indexing="ij"), dim=-1).reshape(-1, 2)
        diff = (pos[:, None, :] - pos[None, :, :]).abs()
        diff = torch.minimum(diff, side - diff)  # torus distance per sheet axis
        self.register_buffer("dist2", (diff.float() ** 2).sum(-1), persistent=False)
        # persistent, so the trained σ is saved with the weights and restored on load
        self.register_buffer("sigma", torch.tensor(side / 2.0))

    def set_sigma(self, sigma: float) -> None:
        self.sigma.fill_(max(float(sigma), self.sigma_floor))

    def kernel(self) -> torch.Tensor:
        k = torch.exp(-self.dist2 / (2 * self.sigma.float() ** 2))
        return k / k.sum(dim=1, keepdim=True)

    def forward(self, x):
        if x is None:
            return None
        k = self.kernel().to(dtype=x.dtype)
        pooled = torch.einsum("ij,bj...->bi...", k, x)
        return pooled if self.mix == 1.0 else (1.0 - self.mix) * x + self.mix * pooled

"""
Fixed topographic recurrence ("Rq" in the lateral/top-down deep dive).

The same channel-sheet Gaussian as SheetPool (Qian et al. 2026), used as recurrence instead of
feedforward pooling:

    recurrence(r) = g · K r        so that   r ← r + α (D + g·K·r − r)

K has rows summing to 1, so the loop gain is at most g. With g = 0.3 the per-area gain stays
inside the cascade rule (≲ 0.3 at T = 16). Nothing is learned here: g and K are fixed, and σ
follows the same shrinking schedule as SheetPool (DyRCNN._update_sheetpool_sigma).
"""

import torch

from dynvision.model_components.recurrence import RecurrenceBase
from dynvision.model_components.sheet_pool import SheetPool

__all__ = ["SheetRecurrence"]


class SheetRecurrence(RecurrenceBase):
    def __init__(
        self,
        in_channels: int,
        gain: float = 0.3,
        sigma_floor: float = 0.75,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.pool = SheetPool(in_channels, sigma_floor=sigma_floor)
        self.register_buffer("gain", torch.tensor(float(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain.to(x.dtype) * self.pool(x)

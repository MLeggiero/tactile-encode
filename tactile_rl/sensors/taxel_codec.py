"""The TaxelScan fast-profile codec in torch, integer-exact with tactile_sim.sensors.taxel_codec (the reference; tests
check both against the frozen golden frames): taxel forces in integer millinewtons, mu-law codes from the shared
threshold table (a code is the number of thresholds at or below the force), and the six patch features (force sum,
centre of pressure in Q8.8 grid units, contact area, peak, change of the sum since the previous frame), every
division rounding half away from zero."""

from __future__ import annotations

import numpy as np
import torch

from tactile_sim.sensors.taxel_codec import CodecSpec


class TorchCodec:
    def __init__(self, spec: CodecSpec, device: str | torch.device = "cpu"):
        self.spec = spec
        self.levels = spec.levels
        self.contact_mn = int(spec.contact_mn)
        self.thresholds = torch.as_tensor(spec.thresholds_mn.astype(np.int64), device=device)
        self.code_force = torch.as_tensor(spec.code_force(np.arange(spec.levels)), dtype=torch.float64, device=device)

    def encode(self, f_mn: torch.Tensor) -> torch.Tensor:
        """Codes (uint8) for taxel forces in integer mN (int64)."""
        return torch.bucketize(f_mn, self.thresholds, right=True).to(torch.uint8)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Representative force (N, float64) of each code."""
        return self.code_force[codes.long()]


def to_mn(force_n: torch.Tensor) -> torch.Tensor:
    """Forces (N) as the firmware holds them: integer mN, rounded half to even like numpy, clipped to [0, 65535]."""
    return torch.clamp(torch.round(force_n.double() * 1000.0), 0.0, 65535.0).long()


def div_round(num: torch.Tensor, den: torch.Tensor | int) -> torch.Tensor:
    """Integer division rounding half away from zero (int64; den > 0)."""
    q = torch.div(num.abs() * 2 + den, 2 * den, rounding_mode="floor")
    return torch.where(num >= 0, q, -q)


def patch_features(f_mn: torch.Tensor, rows: int, cols: int, prev_sum_dN: torch.Tensor,
                   contact_mn: int = 50) -> torch.Tensor:
    """The six features (int64, taxel_codec.FEATURE_FIELDS order) of patches whose taxels (mN, int64) are the last
    dimension of `f_mn` (..., rows * cols), row-major; `prev_sum_dN` (...) is each patch's previous force sum."""
    f = f_mn.long()
    total = f.sum(-1)
    sum_dn = torch.clamp(div_round(total, 100), max=65535)
    r = torch.arange(rows, device=f.device).repeat_interleave(cols)
    c = torch.arange(cols, device=f.device).repeat(rows)
    den = total.clamp(min=1)
    cop_r = torch.where(total > 0, div_round(256 * (f * r).sum(-1), den), torch.full_like(total, -1))
    cop_c = torch.where(total > 0, div_round(256 * (f * c).sum(-1), den), torch.full_like(total, -1))
    area = (f >= contact_mn).sum(-1)
    peak = torch.clamp(div_round(f.max(-1).values, 10), max=65535)
    rate = torch.clamp(sum_dn - prev_sum_dN.long(), -32768, 32767)
    return torch.stack([sum_dn, cop_r, cop_c, area, peak, rate], dim=-1)

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .gaussians import build_covariance


@dataclass
class GaussianAdapterCfg:
    sh_degree: int
    clamping: float = 0.3
    scale_weight: float = 0.001
    isotrophic_covariance: bool = False
    opacity_min: float = 0.0


@dataclass
class AdaptedGaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


class UnifiedGaussianAdapter(nn.Module):
    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

    def forward(
        self,
        means: Float[Tensor, "*batch 3"],
        opacities: Float[Tensor, " *batch"],
        raw_gaussians: Float[Tensor, "*batch _"],
        eps: float = 1e-8,
    ) -> AdaptedGaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        scales = self.cfg.scale_weight * F.softplus(scales)
        if self.cfg.clamping > 0:
            scales = scales.clamp_max(self.cfg.clamping)
        if self.cfg.isotrophic_covariance:
            scales = scales.mean(dim=-1, keepdim=True).expand_as(scales)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        opacities = torch.max(opacities, torch.tensor(self.cfg.opacity_min, device=opacities.device))
        return AdaptedGaussians(
            means,
            build_covariance(scales, rotations),
            sh,
            opacities,
        )

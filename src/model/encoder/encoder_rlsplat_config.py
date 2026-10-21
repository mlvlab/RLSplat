from dataclasses import dataclass
from typing import Literal

from .backbone import BackboneCfg
from .common.gaussian_adapter import GaussianAdapterCfg


@dataclass
class EncoderRLSplatCfg:
    name: Literal["rlsplat"]
    backbone: BackboneCfg
    gaussian_adapter: GaussianAdapterCfg
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    camera_mode: Literal["gt_scaled", "predicted"] = "gt_scaled"
    decoder_depth: int = 2
    max_gaussians_per_anchor: int = 32
    max_gaussian_budget_per_view: int = 16384
    min_gaussian_budget_total: int = 2048
    gaussian_allocation_budget_conditioning: bool = True
    gaussian_allocation_remainder_jitter: bool = False
    position_topk_anchor_count: int = 16
    position_topk_residual_scale: float = 0.1
    position_topk_anchor_attention: bool = False
    position_topk_anchor_attention_first: bool = False
    position_topk_geometry_ffn: bool = True
    position_topk_geometry_attention: bool = True
    position_topk_anchor_include_self: bool = True
    position_topk_geometry_anchor_count: int = 16
    position_topk_geometry_neighbor_count: int = 16
    position_topk_geometry_include_self: bool = True

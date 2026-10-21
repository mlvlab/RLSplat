from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    original_image_shape: list[int]
    input_image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    view_sampler: ViewSamplerCfg
    context_upscale_shape: list[int] | None

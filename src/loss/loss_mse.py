from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..model.decoder.decoder import DecoderOutput
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        target_image: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, ""]:
        delta = prediction.color - target_image
        return self.cfg.weight * (delta**2).mean()

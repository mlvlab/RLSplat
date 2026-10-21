from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..model.decoder.decoder import DecoderOutput

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction: DecoderOutput,
        target_image: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, ""]:
        raise NotImplementedError

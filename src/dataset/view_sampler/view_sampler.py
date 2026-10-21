from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ..types import Stage

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    cameras_are_circular: bool

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        cameras_are_circular: bool,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.cameras_are_circular = cameras_are_circular

    @abstractmethod
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
        Float[Tensor, " overlap"],
    ]:
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

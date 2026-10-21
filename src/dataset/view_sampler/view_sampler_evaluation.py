import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ..types import Stage
from .additional_view_hack import add_addtional_context_index
from .view_sampler import ViewSampler


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]
    overlap: str | float | None = None


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        cameras_are_circular: bool,
    ) -> None:
        super().__init__(cfg, stage, cameras_are_circular)

        self.cfg = cfg

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }

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
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        v = self.num_context_views
        if  v >= 3 and v > len(context_indices):
            context_indices = add_addtional_context_index(context_indices, v)

        return context_indices, target_indices, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0

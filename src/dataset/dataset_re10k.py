import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    relative_pose: bool
    skip_bad_shape: bool
    target_image_shape: list[int] | None = None
    overfit_to_scene: str | None = None
    overfit_to_scenes: list[str] | None = None


@dataclass
class DatasetRE10kCfgWrapper:
    re10k: DatasetRE10kCfg


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    overfit_scene_keys: list[str] | None
    overfit_scene_keys_by_chunk: dict[Path, list[str]]
    overfit_examples_by_chunk: dict[Path, list[dict]]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        if stage != "test":
            raise ValueError("This release provides test datasets only.")
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.overfit_scene_keys = self.get_overfit_scene_keys()
        self.overfit_scene_keys_by_chunk = {}
        self.overfit_examples_by_chunk = {}

        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scenes is not None:
            assert self.overfit_scene_keys is not None
            for scene_key in self.overfit_scene_keys:
                chunk_path = self.index[scene_key]
                self.overfit_scene_keys_by_chunk.setdefault(chunk_path, []).append(
                    scene_key
                )
            selected_chunks = list(self.overfit_scene_keys_by_chunk)
            num_repeats = max(
                1, (len(self.chunks) + len(selected_chunks) - 1) // len(selected_chunks)
            )
            self.chunks = selected_chunks * num_repeats
        elif self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def get_overfit_scene_keys(self) -> list[str] | None:
        if (
            self.cfg.overfit_to_scene is not None
            and self.cfg.overfit_to_scenes is not None
        ):
            raise ValueError(
                "Only one of overfit_to_scene and overfit_to_scenes may be set."
            )

        if self.cfg.overfit_to_scenes is not None:
            scene_keys = self.cfg.overfit_to_scenes
            if not scene_keys:
                raise ValueError("overfit_to_scenes must contain at least one scene.")
            if len(set(scene_keys)) != len(scene_keys):
                raise ValueError("overfit_to_scenes must not contain duplicates.")
            return scene_keys

        if self.cfg.overfit_to_scene is not None:
            return [self.cfg.overfit_to_scene]

        return None

    def _iter_legacy(self):
        """Yield fixed-order chunks, strided across evaluation workers."""
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            if self.cfg.overfit_to_scenes is not None:
                chunk = self.overfit_examples_by_chunk.get(chunk_path)
                if chunk is None:
                    loaded_chunk = torch.load(chunk_path)
                    scene_keys = self.overfit_scene_keys_by_chunk[chunk_path]
                    examples = {
                        x["key"]: x for x in loaded_chunk if x["key"] in scene_keys
                    }
                    missing_scene_keys = set(scene_keys) - set(examples)
                    if missing_scene_keys:
                        raise ValueError(
                            f"Scenes missing from {chunk_path}: "
                            f"{sorted(missing_scene_keys)}"
                        )
                    chunk = [examples[scene_key] for scene_key in scene_keys]
                    self.overfit_examples_by_chunk[chunk_path] = chunk
            else:
                chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            for example in chunk:
                yield example, None

    def __iter__(self):
        for example, _ in self._iter_legacy():
            extrinsics, intrinsics = self.convert_poses(example["cameras"])
            scene = example["key"]

            try:
                context_indices, target_indices, overlap = self.view_sampler.sample(
                    scene,
                    extrinsics,
                    intrinsics,
                )
            except ValueError:
                continue

            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                continue

            try:
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
            except IndexError:
                continue
            except OSError:
                print(f"Skipped bad example {example['key']}.")
                continue

            context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
            target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
            if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                print(
                    f"Skipped bad example {example['key']}. Context shape was "
                    f"{context_images.shape} and target shape was "
                    f"{target_images.shape}."
                )
                continue

            context_extrinsics = extrinsics[context_indices]
            if self.cfg.make_baseline_1:
                a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    print(
                        f"Skipped {scene} because of baseline out of range: "
                        f"{scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            if self.cfg.relative_pose:
                extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / scale,
                    "far": self.get_bound("far", len(context_indices)) / scale,
                    "index": context_indices,
                    "overlap": overlap,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                },
                "scene": scene,
            }
            example = apply_crop_shim(
                example,
                tuple(self.cfg.input_image_shape),
                target_shape=(
                    tuple(self.cfg.target_image_shape)
                    if self.cfg.target_image_shape is not None
                    else None
                ),
            )
            if self.cfg.context_upscale_shape is not None:
                example["context"]["image"] = F.interpolate(
                    example["context"]["image"],
                    size=tuple(self.cfg.context_upscale_shape),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
            yield example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],
        Float[Tensor, "batch 3 3"],
    ]:
        b, _ = poses.shape

        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.overfit_scene_keys is not None:
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.overfit_scene_keys is not None:
            data_stages = ("test",)
        for data_stage in data_stages:
            for root in self.cfg.roots:
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                assert not (set(merged_index.keys()) & set(index.keys()))

                merged_index = {**merged_index, **index}
        return merged_index

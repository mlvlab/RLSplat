import os
from dataclasses import dataclass
from types import MethodType
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ....dataset.types import BatchedViews
from ...camera_prediction import PredictedCameras, dvlt_pose_encoding_to_cameras
from .backbone import Backbone


def _identity_torch_compile(model=None, *args, **kwargs):
    """Return the input callable unchanged, matching torch.compile call styles."""
    del args, kwargs
    if model is None:
        return lambda fn: fn
    return model


def _initialize_accelerate_logging() -> None:
    from accelerate import PartialState

    distributed_env_names = (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
    )
    distributed_env = {
        name: os.environ.pop(name)
        for name in distributed_env_names
        if name in os.environ
    }
    try:
        PartialState(cpu=True)
    finally:
        os.environ.update(distributed_env)


@dataclass
class BackboneDVLTCfg:
    name: Literal["dvlt_multi"]
    img_size: int = 504
    patch_size: int = 14
    embed_dim: int = 768
    num_steps: int = 16
    min_steps: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    patch_embed: Literal[
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vits14_reg",
        "dinov2_vitg2_reg",
    ] = "dinov2_vitb14_reg"
    load_patch_embed_weights: bool = False
    decoder_depth: int = 2
    decoder_embed_dim: int = 384
    decoder_num_heads: int = 6
    camera_head: bool = True
    recurrence_mode: Literal["gated", "no_sout", "no_depthscale", "none"] = "gated"
    time_conditioning: Literal["continuous", "interval"] = "interval"
    k_sampling: Literal["linspace"] | None = "linspace"
    inference_steps: int | None = 12
    decoder_head_type: Literal["linear", "conv"] = "linear"
    depth_head_type: Literal["linear", "conv"] | None = "conv"
    depth_decoder_depth: int | None = None
    depth_decoder_embed_dim: int | None = None
    depth_decoder_num_heads: int | None = None
    decoder_init_values: float | None = None
    decode_chunk_size: int | None = None
    disable_torch_compile: bool = True
    pretrained: str = "nvidia/dvlt"
    strict_load: bool = True
    rope_position_scale: Literal["native", "official"] = "native"


class BackboneDVLT(Backbone[BackboneDVLTCfg]):
    """DVLT feature adapter that matches the VGGT token backbone contract."""

    def __init__(
        self,
        cfg: BackboneDVLTCfg,
        d_in: int,
        predict_cameras: bool = False,
    ) -> None:
        super().__init__(cfg)
        if d_in != 3:
            raise ValueError(f"DVLT expects RGB inputs, got d_in={d_in}")

        original_torch_compile = None
        if cfg.disable_torch_compile and hasattr(torch, "compile"):
            original_torch_compile = torch.compile
            torch.compile = _identity_torch_compile
        try:
            from dvlt.model.dvlt.model import DVLT
        except ImportError as exc:
            raise ImportError(
                "BackboneDVLT requires nv-tlabs/dvlt to be installed. "
                "DVLT declares Python >=3.11; install it in a compatible "
                "environment, for example with "
                "`pip install git+https://github.com/nv-tlabs/dvlt.git`."
            ) from exc
        finally:
            if original_torch_compile is not None:
                torch.compile = original_torch_compile

        dvlt = DVLT(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            num_steps=cfg.num_steps,
            min_steps=cfg.min_steps,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            num_register_tokens=cfg.num_register_tokens,
            patch_embed=cfg.patch_embed,
            load_patch_embed_weights=cfg.load_patch_embed_weights,
            decoder_depth=cfg.decoder_depth,
            decoder_embed_dim=cfg.decoder_embed_dim,
            decoder_num_heads=cfg.decoder_num_heads,
            camera_head=cfg.camera_head,
            recurrence_mode=cfg.recurrence_mode,
            time_conditioning=cfg.time_conditioning,
            k_sampling=cfg.k_sampling,
            inference_steps=cfg.inference_steps,
            decoder_head_type=cfg.decoder_head_type,
            depth_head_type=cfg.depth_head_type,
            depth_decoder_depth=cfg.depth_decoder_depth,
            depth_decoder_embed_dim=cfg.depth_decoder_embed_dim,
            depth_decoder_num_heads=cfg.depth_decoder_num_heads,
            decoder_init_values=cfg.decoder_init_values,
            decode_chunk_size=cfg.decode_chunk_size,
        )
        if cfg.pretrained:
            _initialize_accelerate_logging()
            dvlt.load_pretrained(cfg.pretrained, strict=cfg.strict_load)

        self.model = dvlt.model
        self.predict_cameras = predict_cameras
        self._enable_fractional_rope_positions()
        self._drop_unused_decoder_heads()

    @staticmethod
    def _patch_fractional_rope(rope: nn.Module) -> None:
        if getattr(rope, "_rlsplat_fractional_positions", False):
            return

        def _compute_fractional_components(
            self,
            dim: int,
            positions: Tensor,
            dtype: torch.dtype,
        ) -> tuple[Tensor, Tensor]:
            exponents = torch.arange(0, dim, 2, device=positions.device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            angles = torch.einsum("bn,d->bnd", positions.float(), inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            return angles.cos().to(dtype), angles.sin().to(dtype)

        def _apply_1d_rope(
            self,
            tokens: Tensor,
            positions: Tensor,
            cos_comp: Tensor,
            sin_comp: Tensor,
        ) -> Tensor:
            if positions.dtype.is_floating_point:
                cos, sin = self._compute_fractional_components(
                    tokens.shape[-1],
                    positions,
                    tokens.dtype,
                )
                cos = cos[:, None, :, :]
                sin = sin[:, None, :, :]
            else:
                cos = F.embedding(positions, cos_comp)[:, None, :, :]
                sin = F.embedding(positions, sin_comp)[:, None, :, :]
            return (tokens * cos) + (self._rotate_features(tokens) * sin)

        rope._compute_fractional_components = MethodType(
            _compute_fractional_components,
            rope,
        )
        rope._apply_1d_rope = MethodType(_apply_1d_rope, rope)
        rope._rlsplat_fractional_positions = True

    def _enable_fractional_rope_positions(self) -> None:
        for module in self.model.modules():
            if (
                hasattr(module, "base_frequency")
                and hasattr(module, "_apply_1d_rope")
                and hasattr(module, "_rotate_features")
            ):
                self._patch_fractional_rope(module)

    def _drop_unused_decoder_heads(self) -> None:
        if self.predict_cameras:
            if not hasattr(self.model, "camera_head") or self.model.camera_head is None:
                raise ValueError(
                    "camera_mode='predicted' requires the DVLT camera head"
                )
            if not hasattr(self.model, "ray_decoder"):
                raise ValueError(
                    "camera_mode='predicted' requires the DVLT ray decoder"
                )
            ray_decoder = self.model.ray_decoder
            for name in ("norm", "head", "upsample_blocks", "output_block"):
                if hasattr(ray_decoder, name):
                    delattr(ray_decoder, name)
            if hasattr(self.model, "depth_decoder"):
                delattr(self.model, "depth_decoder")
            return

        for name in ("ray_decoder", "depth_decoder", "camera_head"):
            if hasattr(self.model, name):
                delattr(self.model, name)

    def _predict_context_cameras(
        self,
        tokens: Tensor,
        rope_pos: Tensor,
        batch_size: int,
        num_views: int,
        image_shape: tuple[int, int],
    ) -> PredictedCameras:
        ray_decoder = self.model.ray_decoder
        ray_features = ray_decoder.proj_in(tokens)
        for block in ray_decoder.blocks:
            ray_features = block(ray_features, pos=rope_pos)
        camera_features = ray_features[:, 0]
        pose_encoding = self.model.camera_head(
            camera_features,
            batch_size,
            num_views,
        )
        return dvlt_pose_encoding_to_cameras(
            pose_encoding,
            image_shape,
        )

    def _camera_tokens(self, batch_size: int, num_views: int) -> Tensor:
        camera_token = self.model.camera_token
        first = camera_token[:, 0:1].expand(batch_size, 1, -1, -1)
        rest = camera_token[:, 1:2].expand(batch_size, num_views - 1, -1, -1)
        return torch.cat([first, rest], dim=1).reshape(
            batch_size * num_views,
            *camera_token.shape[2:],
        )

    def _scale_rope_positions_to_official(
        self,
        pos: Tensor,
        patch_height: int,
        patch_width: int,
    ) -> Tensor:
        if self.cfg.rope_position_scale == "native":
            return pos
        if self.cfg.rope_position_scale != "official":
            raise ValueError(f"Unknown rope_position_scale={self.cfg.rope_position_scale}")

        target_patch_size = self.cfg.img_size // self.cfg.patch_size
        target_height = target_patch_size
        target_width = target_patch_size
        if (patch_height, patch_width) == (target_height, target_width):
            return pos

        pos = pos.float()
        pos_y = (pos[..., 0] + 0.5) * (target_height / patch_height) - 0.5
        pos_x = (pos[..., 1] + 0.5) * (target_width / patch_width) - 0.5
        return torch.stack((pos_y, pos_x), dim=-1)

    def _get_rope_positions(
        self,
        batch_size_view: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        patch_height = height // self.cfg.patch_size
        patch_width = width // self.cfg.patch_size
        pos = self.model.position_getter(
            batch_size_view,
            patch_height,
            patch_width,
            device,
        )
        pos = self._scale_rope_positions_to_official(pos, patch_height, patch_width)
        pos = pos + 1
        special = torch.zeros(
            batch_size_view,
            self.model.patch_start_idx,
            2,
            device=device,
            dtype=pos.dtype,
        )
        return torch.cat([special, pos], dim=1)

    def forward(
        self,
        context: BatchedViews,
        symmetrize_batch: bool = False,
        return_views: bool = False,
    ) -> tuple[list[Tensor], Tensor, int, PredictedCameras | None]:
        del symmetrize_batch, return_views

        images = context["image"]
        batch_size, num_views, _, height, width = images.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "DVLT backbone requires image height and width to be divisible "
                f"by patch_size={self.patch_size}, got {(height, width)}."
            )

        z_0 = self.model._encode_images(images)
        register_token = self.model.register_token.expand(
            batch_size,
            num_views,
            -1,
            -1,
        ).reshape(batch_size * num_views, self.model.num_register_tokens, -1)
        camera_token = self._camera_tokens(batch_size, num_views)
        tokens = torch.cat([camera_token, register_token, z_0], dim=1)
        rope_pos = self._get_rope_positions(
            batch_size * num_views,
            height,
            width,
            images.device,
        )

        tokens = self.model._solve_inference(
            tokens,
            rope_pos,
            batch_size,
            num_views,
        )
        cameras = None
        if self.predict_cameras:
            cameras = self._predict_context_cameras(
                tokens,
                rope_pos,
                batch_size,
                num_views,
                (height, width),
            )
        output_tokens = [
            rearrange(tokens, "(b v) n d -> b v n d", b=batch_size, v=num_views)
        ]

        shape_all = torch.tensor(
            [height, width],
            device=images.device,
            dtype=torch.long,
        )[None].repeat(batch_size * num_views, 1)
        shape = rearrange(shape_all, "(b v) c -> b v c", b=batch_size, v=num_views)
        return output_tokens, shape, self.model.patch_start_idx, cameras

    @property
    def patch_size(self) -> int:
        return self.cfg.patch_size

    @property
    def d_out(self) -> int:
        return self.cfg.embed_dim

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians_b
from .cuda_splatting import render_cuda
from .decoder import Decoder, DecoderOutput, DecoderSplattingCUDACfg


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )
        self.low_pass_filter = cfg.low_pass_filter

    def forward(
        self,
        gaussians: list[Gaussians_b],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        camera_pose_grad: bool = False,
    ) -> DecoderOutput:
        _, views, _, _ = extrinsics.shape
        colors, depths = [], []
        for batch_idx, scene in enumerate(gaussians):
            color, depth = render_cuda(
                extrinsics[batch_idx],
                intrinsics[batch_idx],
                near[batch_idx],
                far[batch_idx],
                image_shape,
                repeat(self.background_color, "c -> v c", v=views),
                repeat(scene.means, "g xyz -> v g xyz", v=views),
                repeat(scene.covariances, "g i j -> v g i j", v=views),
                repeat(scene.harmonics, "g c d_sh -> v g c d_sh", v=views),
                repeat(scene.opacities, "g -> v g", v=views),
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=cam_rot_delta[batch_idx] if cam_rot_delta is not None else None,
                cam_trans_delta=cam_trans_delta[batch_idx] if cam_trans_delta is not None else None,
                low_pass_filter=self.low_pass_filter,
                camera_pose_grad=camera_pose_grad,
            )
            colors.append(color)
            depths.append(depth)
        return DecoderOutput(torch.stack(colors), torch.stack(depths), None)

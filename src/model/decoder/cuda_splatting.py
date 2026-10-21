from math import isqrt
from typing import Literal

import gsplat
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ...misc.cam_utils import SE3_exp

NEAR_PLANE = 0.2


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
    low_pass_filter: float = 0.3,
    camera_pose_grad: bool = False,
) -> tuple[
    Float[Tensor, "batch 3 height width"],
    Float[Tensor, "batch height width"],
]:
    """Render with gsplat. `extrinsics` is camera-to-world; `intrinsics` is
    normalized (fx, fy, cx, cy as fractions of width/height, matching
    src/geometry/projection.py), not pixel units.
    """
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    if camera_pose_grad and (cam_rot_delta is not None or cam_trans_delta is not None):
        raise ValueError(
            "camera_pose_grad cannot be combined with explicit camera deltas"
        )

    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    view_matrix = extrinsics.inverse()

    if cam_rot_delta is not None or cam_trans_delta is not None:
        zero = torch.zeros_like(
            cam_rot_delta if cam_rot_delta is not None else cam_trans_delta
        )
        tau = torch.cat(
            [
                cam_trans_delta if cam_trans_delta is not None else zero,
                cam_rot_delta if cam_rot_delta is not None else zero,
            ],
            dim=-1,
        )
        perturbation = torch.stack([SE3_exp(tau[i]) for i in range(tau.shape[0])])
        view_matrix = perturbation @ view_matrix

    h, w = image_shape

    pixel_scale = extrinsics.new_tensor([w, h, 1.0])[:, None]
    camera_matrix = intrinsics * pixel_scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    render_colors, _render_alphas, _meta = gsplat.rasterization(
        means=gaussian_means,
        quats=None,
        scales=None,
        covars=gaussian_covariances,
        opacities=gaussian_opacities,
        colors=shs if use_sh else shs[:, :, 0, :],
        viewmats=view_matrix[:, None],
        Ks=camera_matrix[:, None],
        width=w,
        height=h,
        near_plane=NEAR_PLANE,
        eps2d=low_pass_filter,
        sh_degree=degree if use_sh else None,
        backgrounds=background_color[:, None, :],
        render_mode="RGB+D",
        packed=False,
    )
    render_colors = render_colors[:, 0]
    image = rearrange(render_colors[..., :3], "b h w c -> b c h w")
    depth = render_colors[..., 3]

    return image, depth


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

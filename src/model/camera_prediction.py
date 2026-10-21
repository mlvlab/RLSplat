from dataclasses import dataclass

import torch
from torch import Tensor

from ..dataset.types import BatchedExample


@dataclass
class PredictedCameras:
    """DVLT camera predictions in RLSplat's camera conventions."""

    pose_encoding: Tensor
    extrinsics: Tensor
    intrinsics: Tensor


@dataclass
class ResolvedCameras:
    target_extrinsics: Tensor
    target_intrinsics: Tensor
    target_near: Tensor
    target_far: Tensor
    context_extrinsics: Tensor
    context_intrinsics: Tensor
    context_near: Tensor
    context_far: Tensor
    target_scale: Tensor
    context_scale: Tensor


def normalized_to_pixel_intrinsics(
    intrinsics: Tensor,
    image_shape: tuple[int, int],
) -> Tensor:
    height, width = image_shape
    pixel_intrinsics = intrinsics.clone()
    pixel_intrinsics[..., 0, :] *= float(width)
    pixel_intrinsics[..., 1, :] *= float(height)
    return pixel_intrinsics


def pixel_to_normalized_intrinsics(
    intrinsics: Tensor,
    image_shape: tuple[int, int],
) -> Tensor:
    height, width = image_shape
    normalized_intrinsics = intrinsics.clone()
    normalized_intrinsics[..., 0, :] /= float(width)
    normalized_intrinsics[..., 1, :] /= float(height)
    return normalized_intrinsics


def dvlt_pose_encoding_to_cameras(
    pose_encoding: Tensor,
    image_shape: tuple[int, int],
) -> PredictedCameras:
    from dvlt.model_components.pose_encoding import pose_enc_to_extri_intri

    extrinsics_3x4, pixel_intrinsics = pose_enc_to_extri_intri(
        pose_encoding,
        images_hw=image_shape,
    )
    if pixel_intrinsics is None:
        raise ValueError("DVLT predicted-camera mode requires 9D pose encodings")
    if extrinsics_3x4.shape[-2:] != (3, 4):
        raise ValueError(
            "DVLT pose decoding must return c2w matrices with shape [..., 3, 4], "
            f"got {tuple(extrinsics_3x4.shape)}"
        )

    bottom_row = extrinsics_3x4.new_tensor([0.0, 0.0, 0.0, 1.0])
    bottom_row = bottom_row.expand(*extrinsics_3x4.shape[:-2], 1, 4)
    extrinsics = torch.cat([extrinsics_3x4, bottom_row], dim=-2)
    intrinsics = pixel_to_normalized_intrinsics(pixel_intrinsics, image_shape)
    return PredictedCameras(pose_encoding, extrinsics, intrinsics)


def _require_context_baselines(
    gt_context_extrinsics: Tensor,
    epsilon: float = 1e-8,
) -> None:
    if gt_context_extrinsics.shape[1] < 2:
        raise ValueError("predicted camera mode requires at least two context views")
    gt_centers = gt_context_extrinsics[..., :3, 3]
    reference_distances = (gt_centers[:, 1:] - gt_centers[:, :1]).norm(dim=-1)
    if not bool((reference_distances.sum(dim=1) > epsilon).all()):
        raise ValueError(
            "predicted camera mode requires a non-degenerate GT context baseline"
        )


def first_context_baseline_scale(
    gt_context_extrinsics: Tensor,
    pred_context_extrinsics: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    _require_context_baselines(gt_context_extrinsics, epsilon)
    gt_centers = gt_context_extrinsics[..., :3, 3]
    pred_centers = pred_context_extrinsics[..., :3, 3]
    gt_scale = (gt_centers[:, 1:] - gt_centers[:, :1]).norm(dim=-1).sum(dim=1)
    pred_scale = (
        (pred_centers[:, 1:] - pred_centers[:, :1]).norm(dim=-1).sum(dim=1)
    )
    if not bool((pred_scale > epsilon).all()):
        raise ValueError(
            "predicted camera mode requires a non-degenerate predicted "
            "context baseline"
        )
    return pred_scale / gt_scale.clamp_min(epsilon)


def align_gt_context_to_predicted_scale(
    gt_context_extrinsics: Tensor,
    pred_context_extrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        scale = first_context_baseline_scale(
            gt_context_extrinsics,
            pred_context_extrinsics.detach(),
        )
        aligned = gt_context_extrinsics.clone()
        aligned[..., :3, 3] *= scale[:, None, None]
    return aligned, scale


def compute_dvlt_camera_loss(
    predicted: PredictedCameras,
    gt_context_extrinsics: Tensor,
    gt_context_intrinsics: Tensor,
    image_shape: tuple[int, int],
) -> tuple[Tensor, dict[str, Tensor]]:
    from dvlt.model_components.pose_encoding import extri_intri_to_pose_enc

    aligned_gt_extrinsics, baseline_scale = align_gt_context_to_predicted_scale(
        gt_context_extrinsics,
        predicted.extrinsics,
    )
    with torch.no_grad():
        gt_pose_encoding = extri_intri_to_pose_enc(
            aligned_gt_extrinsics[..., :3, :4],
            normalized_to_pixel_intrinsics(gt_context_intrinsics, image_shape),
            image_shape,
        )

    pred_pose_encoding = predicted.pose_encoding.float()
    gt_pose_encoding = gt_pose_encoding.to(pred_pose_encoding)
    loss_translation = (
        pred_pose_encoding[..., :3] - gt_pose_encoding[..., :3]
    ).abs().clamp(max=100.0).mean()
    loss_rotation = (
        pred_pose_encoding[..., 3:7] - gt_pose_encoding[..., 3:7]
    ).abs().mean()
    loss_fov = (
        pred_pose_encoding[..., 7:] - gt_pose_encoding[..., 7:]
    ).abs().mean()
    loss = loss_translation + loss_rotation + 0.5 * loss_fov

    pred_rotation = predicted.extrinsics[..., :3, :3].float()
    gt_rotation = aligned_gt_extrinsics[..., :3, :3].float()
    relative_rotation = pred_rotation.transpose(-1, -2) @ gt_rotation
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_degrees = torch.rad2deg(
        torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    ).mean()
    translation_l2 = (
        predicted.extrinsics[..., :3, 3].float()
        - aligned_gt_extrinsics[..., :3, 3].float()
    ).norm(dim=-1).mean()
    return loss, {
        "translation": loss_translation.detach(),
        "rotation": loss_rotation.detach(),
        "fov": loss_fov.detach(),
        "rotation_degrees": rotation_degrees.detach(),
        "translation_l2": translation_l2.detach(),
        "baseline_scale": baseline_scale.mean().detach(),
    }


def _gather_views(values: Tensor, indices: Tensor) -> Tensor:
    trailing_shape = values.shape[2:]
    gather_index = indices[(...,) + (None,) * len(trailing_shape)]
    gather_index = gather_index.expand(*indices.shape, *trailing_shape)
    return values.gather(1, gather_index)


def _align_target_cameras(
    gt_context_extrinsics: Tensor,
    pred_context_extrinsics: Tensor,
    gt_target_extrinsics: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _require_context_baselines(gt_context_extrinsics)
    gt_context_centers = gt_context_extrinsics[..., :3, 3]
    pred_context_centers = pred_context_extrinsics[..., :3, 3]
    gt_target_centers = gt_target_extrinsics[..., :3, 3]

    with torch.no_grad():
        nearest_context = torch.cdist(
            gt_target_centers.float(),
            gt_context_centers.float(),
        ).argmin(dim=-1)
    gt_anchor = _gather_views(gt_context_extrinsics, nearest_context)
    pred_anchor = _gather_views(pred_context_extrinsics, nearest_context)

    gt_anchor_centers = gt_anchor[..., :3, 3]
    pred_anchor_centers = pred_anchor[..., :3, 3]
    gt_baselines = (
        gt_context_centers[:, None] - gt_anchor_centers[:, :, None]
    ).norm(dim=-1)
    pred_baselines = (
        pred_context_centers[:, None] - pred_anchor_centers[:, :, None]
    ).norm(dim=-1)
    valid_baselines = gt_baselines > 1e-8
    if not bool(valid_baselines.any(dim=-1).all()):
        raise ValueError(
            "target alignment requires a non-degenerate context baseline"
        )
    if not bool(
        ((pred_baselines * valid_baselines).sum(dim=-1) > 1e-8).all()
    ):
        raise ValueError(
            "target alignment requires a non-degenerate predicted "
            "context baseline"
        )
    scale = (
        (pred_baselines * valid_baselines).sum(dim=-1)
        / (gt_baselines * valid_baselines).sum(dim=-1).clamp_min(1e-8)
    )

    target_relative = torch.linalg.inv(gt_anchor.float()) @ gt_target_extrinsics.float()
    scaled_relative = target_relative.clone()
    scaled_relative[..., :3, 3] = target_relative[..., :3, 3] * scale[..., None]
    aligned_target = pred_anchor.float() @ scaled_relative
    return aligned_target.to(pred_context_extrinsics.dtype), nearest_context, scale


def _align_target_intrinsics(
    gt_context_intrinsics: Tensor,
    pred_context_intrinsics: Tensor,
    gt_target_intrinsics: Tensor,
    nearest_context: Tensor,
) -> Tensor:
    gt_anchor = _gather_views(gt_context_intrinsics, nearest_context)
    pred_anchor = _gather_views(pred_context_intrinsics, nearest_context)
    aligned = pred_anchor.clone()
    for row, column in ((0, 0), (1, 1), (0, 2), (1, 2)):
        ratio = gt_target_intrinsics[..., row, column] / gt_anchor[
            ..., row, column
        ].clamp_min(1e-8)
        aligned[..., row, column] = pred_anchor[..., row, column] * ratio
    return aligned


def resolve_cameras(
    batch: BatchedExample,
    predicted: PredictedCameras | None,
) -> ResolvedCameras:
    if predicted is None:
        target_scale = torch.ones_like(batch["target"]["near"])
        context_scale = torch.ones(
            batch["context"]["near"].shape[0],
            device=batch["context"]["near"].device,
            dtype=batch["context"]["near"].dtype,
        )
        return ResolvedCameras(
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            target_scale,
            context_scale,
        )

    gt_context_extrinsics = batch["context"]["extrinsics"].to(
        predicted.extrinsics
    )
    gt_target_extrinsics = batch["target"]["extrinsics"].to(
        predicted.extrinsics
    )
    target_extrinsics, nearest_context, target_scale = _align_target_cameras(
        gt_context_extrinsics,
        predicted.extrinsics,
        gt_target_extrinsics,
    )
    target_intrinsics = _align_target_intrinsics(
        batch["context"]["intrinsics"].to(predicted.intrinsics),
        predicted.intrinsics,
        batch["target"]["intrinsics"].to(predicted.intrinsics),
        nearest_context,
    )
    context_scale = first_context_baseline_scale(
        gt_context_extrinsics,
        predicted.extrinsics,
    )
    target_bound_scale = target_scale.detach().to(batch["target"]["near"])
    context_bound_scale = context_scale.detach().to(batch["context"]["near"])
    return ResolvedCameras(
        target_extrinsics,
        target_intrinsics,
        batch["target"]["near"] * target_bound_scale,
        batch["target"]["far"] * target_bound_scale,
        predicted.extrinsics,
        predicted.intrinsics,
        batch["context"]["near"] * context_bound_scale[:, None],
        batch["context"]["far"] * context_bound_scale[:, None],
        target_scale,
        context_scale,
    )

"""Test-time evaluation of ModelWrapper: the per-scene metric/visualization
pass and the optional pose alignment. Split out of model_wrapper.py; the
wrapper's test_step / test_step_align delegate here."""
import torch
from einops import rearrange
from torch import nn
from tqdm import tqdm

from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..misc.cam_utils import update_pose
from ..misc.image_io import save_image
from .camera_prediction import (
    compute_dvlt_camera_loss,
    resolve_cameras,
)
from .validation_budgets import (
    get_num_gaussians_for_log,
    normalized_budget_auc,
    resolve_test_budgets,
)


def run_test_step(module, batch, batch_idx):
    """Body of ModelWrapper.test_step (``module`` is the wrapper)."""
    batch: BatchedExample = module.data_shim(batch)

    b, v, _, h, w = batch["target"]["image"].shape

    assert b == 1
    if batch_idx % 100 == 0:
        print(f"Test step {batch_idx:0>6}.")

    num_anchor_tokens = 1
    num_context_views = 1
    if module.test_cfg.gaussian_budget_sweep:
        num_context_views = batch["context"]["image"].shape[1]
        context_height, context_width = batch["context"]["image"].shape[-2:]
        num_anchor_tokens = (
            num_context_views
            * (context_height // module.encoder.patch_size)
            * (context_width // module.encoder.patch_size)
        )
    test_budgets, primary_budget = resolve_test_budgets(
        module.encoder,
        num_anchor_tokens,
        module.test_cfg.gaussian_budget_per_batch,
        module.test_cfg.gaussian_budget_sweep,
        num_context_views,
        per_view_list=module.test_cfg.gaussian_budget_per_view_list,
    )
    test_backbone_cache = (
        {} if module.test_cfg.gaussian_budget_sweep else None
    )
    rgb_gt = batch["target"]["image"][0]
    primary_result = None
    budget_metric_rows = {}
    budget_metric_means = {
        "lpips": {},
        "ssim": {},
        "psnr": {},
    }
    evaluation_budgets = [
        budget for budget in test_budgets if budget != primary_budget
    ] + [primary_budget]

    for budget in evaluation_budgets:
        strategy_dump = {}
        encoder_kwargs = {
            "visualization_dump": strategy_dump,
            "gaussian_budget_per_batch": budget,
        }
        if test_backbone_cache is not None:
            encoder_kwargs["validation_backbone_cache"] = (
                test_backbone_cache
            )

        with module.benchmarker.time("encoder"):
            gaussians_i, predicted_cameras_i = module._encode_with_cameras(
                batch["context"],
                **encoder_kwargs,
            )
        cameras_i = resolve_cameras(batch, predicted_cameras_i)

        if module.test_cfg.align_pose:
            output_i, render_extrinsics_i = module.test_step_align(
                batch,
                gaussians_i,
                cameras_i.target_extrinsics,
                cameras_i.target_intrinsics,
                cameras_i.target_near,
                cameras_i.target_far,
                verbose=budget == primary_budget,
            )
        else:
            render_extrinsics_i = cameras_i.target_extrinsics
            with module.benchmarker.time("decoder", num_calls=v):
                output_i = module.decoder.forward(
                    gaussians_i,
                    cameras_i.target_extrinsics,
                    cameras_i.target_intrinsics,
                    cameras_i.target_near,
                    cameras_i.target_far,
                    (h, w),
                )

        result_i = {
            "gaussians": gaussians_i,
            "predicted_cameras": predicted_cameras_i,
            "cameras": cameras_i,
            "output": output_i,
            "strategy_dump": strategy_dump,
            "render_extrinsics": render_extrinsics_i,
        }
        if module.test_cfg.compute_scores:
            rgb_pred_i = output_i.color[0]
            metrics_i = {
                "lpips": compute_lpips(rgb_gt, rgb_pred_i),
                "ssim": compute_ssim(rgb_gt, rgb_pred_i),
                "psnr": compute_psnr(rgb_gt, rgb_pred_i),
            }
            result_i["metrics"] = metrics_i
            num_gaussians_i = get_num_gaussians_for_log(gaussians_i)
            result_i["num_gaussians"] = num_gaussians_i
            gaussian_counts_i = strategy_dump.get("gaussian_counts", None)
            budget_metric_row = {
                metric_name: metric_values.detach().cpu().tolist()
                for metric_name, metric_values in metrics_i.items()
            }
            budget_metric_row["num_gaussians"] = float(
                num_gaussians_i.detach().cpu()
            )
            if gaussian_counts_i is not None:
                gaussian_counts_i_f = gaussian_counts_i.detach().float()
                budget_metric_row["gaussian_counts"] = {
                    "mean": float(gaussian_counts_i_f.mean().cpu()),
                    "std": float(
                        gaussian_counts_i_f.std(unbiased=False).cpu()
                    ),
                    "min": float(gaussian_counts_i_f.min().cpu()),
                    "max": float(gaussian_counts_i_f.max().cpu()),
                }
            budget_metric_rows[budget] = budget_metric_row
            for metric_name, metric_values in metrics_i.items():
                metric_mean = metric_values.mean()
                budget_metric_means[metric_name][budget] = metric_mean
                if module.test_cfg.gaussian_budget_sweep:
                    module.log(
                        f"test_{budget}/metrics/{metric_name}",
                        metric_mean,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=1,
                    )
            if module.test_cfg.gaussian_budget_sweep:
                module.log(
                    f"test_{budget}/allocation/num_gaussians",
                    num_gaussians_i,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=1,
                )

        if budget == primary_budget:
            primary_result = result_i
        else:
            del (
                gaussians_i,
                predicted_cameras_i,
                cameras_i,
                output_i,
                strategy_dump,
                result_i,
            )
            if module.test_cfg.compute_scores:
                del rgb_pred_i, metrics_i, num_gaussians_i

    if primary_result is None:
        raise RuntimeError("The primary test budget was not evaluated")

    predicted_cameras = primary_result["predicted_cameras"]
    output = primary_result["output"]
    strategy_dump = primary_result["strategy_dump"]

    if module.test_cfg.compute_scores:
        primary_metrics = primary_result["metrics"]
        lpips_per_target = primary_metrics["lpips"]
        ssim_per_target = primary_metrics["ssim"]
        psnr_per_target = primary_metrics["psnr"]
        psnr = psnr_per_target.mean()
        all_metrics = {
            "lpips_ours": lpips_per_target.mean(),
            "ssim_ours": ssim_per_target.mean(),
            "psnr_ours": psnr,
        }
        module.log_dict(
            all_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        gaussian_counts = strategy_dump.get("gaussian_counts", None)
        row = {
            "scene": batch["scene"][0],
            "context_indices": batch["context"]["index"][0].tolist(),
            "target_indices": batch["target"]["index"][0].tolist(),
            "lpips": lpips_per_target.detach().cpu().tolist(),
            "ssim": ssim_per_target.detach().cpu().tolist(),
            "psnr": psnr_per_target.detach().cpu().tolist(),
            "num_gaussians": float(
                primary_result["num_gaussians"].detach().cpu()
            ),
        }
        if primary_budget is not None:
            row["primary_budget"] = primary_budget
        if gaussian_counts is not None:
            gaussian_counts_f = gaussian_counts.detach().float()
            row["gaussian_counts"] = {
                "mean": float(gaussian_counts_f.mean().cpu()),
                "std": float(
                    gaussian_counts_f.std(unbiased=False).cpu()
                ),
                "min": float(gaussian_counts_f.min().cpu()),
                "max": float(gaussian_counts_f.max().cpu()),
            }
        if predicted_cameras is not None:
            _, camera_metrics = compute_dvlt_camera_loss(
                predicted_cameras,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["image"].shape[-2:],
            )
            row["camera"] = {
                name: float(value.detach().cpu())
                for name, value in camera_metrics.items()
            }

        if module.test_cfg.gaussian_budget_sweep:
            row["budget_metrics"] = {
                str(budget): budget_metric_rows[budget]
                for budget in test_budgets
            }

            budget_auc = {
                metric_name: normalized_budget_auc(
                    test_budgets,
                    [
                        metric_values[budget]
                        for budget in test_budgets
                    ],
                )
                for metric_name, metric_values in budget_metric_means.items()
            }
            row["budget_auc"] = {
                metric_name: float(metric_value.detach().cpu())
                for metric_name, metric_value in budget_auc.items()
            }
            for metric_name, metric_value in budget_auc.items():
                module.log(
                    f"test/auc_normalized/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=1,
                )
        module.test_metric_rows.append(row)

    (scene,) = batch["scene"]
    name = get_cfg()["wandb"]["name"]
    path = module.test_cfg.output_path / name
    if module.test_cfg.save_image:
        for index, color in zip(batch["target"]["index"][0], output.color[0]):
            save_image(color, path / scene / f"color/{index:0>6}.png")


@torch.no_grad()
def run_test_step_align(
    module,
    batch,
    gaussians,
    initial_extrinsics,
    intrinsics,
    near,
    far,
    verbose=False,
):
    module.encoder.eval()
    for param in module.encoder.parameters():
        param.requires_grad = False

    b, v, _, h, w = batch["target"]["image"].shape

    if not module.losses:
        raise ValueError("test-time pose alignment needs at least one loss")

    with torch.set_grad_enabled(True):
        cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=module.device))
        cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=module.device))

        opt_params = []
        opt_params.append(
            {
                "params": [cam_rot_delta],
                "lr": module.test_cfg.rot_opt_lr,
            }
        )
        opt_params.append(
            {
                "params": [cam_trans_delta],
                "lr": module.test_cfg.trans_opt_lr,
            }
        )
        pose_optimizer = torch.optim.Adam(opt_params)

        extrinsics = initial_extrinsics.clone()

        if verbose:
            logger = tqdm(range(module.test_cfg.pose_align_steps))
        else:
            logger = range(module.test_cfg.pose_align_steps)

        prev_loss = None
        patience_counter = 0
        patience_limit = 10 

        with module.benchmarker.time("optimize"):
            for i in logger:
                pose_optimizer.zero_grad()

                output = module.decoder.forward(
                    gaussians,
                    extrinsics,
                    intrinsics,
                    near,
                    far,
                    (h, w),
                    cam_rot_delta=cam_rot_delta,
                    cam_trans_delta=cam_trans_delta,
                )

                total_loss = 0


                for loss_fn in module.losses:
                    loss = loss_fn.forward(
                        output,
                        batch["target"]["image"],
                    )
                    total_loss = total_loss + loss

                if verbose:
                    logger.set_description(f"pose optim step {i}; loss = {total_loss:.6f}")

                total_loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                )
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)

                    extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

                if prev_loss is not None:
                    delta = abs(total_loss.item() - prev_loss)
                    if delta < 0.00001:
                        patience_counter += 1
                        if patience_counter >= patience_limit and i >= 100:
                            break
                    else:
                        patience_counter = 0
                prev_loss = total_loss.item()


    with module.benchmarker.time("decoder", num_calls=v):
        output = module.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
        )
    del pose_optimizer

    return output, extrinsics

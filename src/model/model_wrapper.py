"""Evaluation module: rendering, camera alignment, metrics and local outputs."""
import json

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from ..dataset.data_module import get_data_shim
from ..global_cfg import get_cfg
from ..misc.benchmarker import Benchmarker
from .camera_prediction import PredictedCameras
from .model_wrapper_test_step import run_test_step, run_test_step_align


class ModelWrapper(LightningModule):
    def __init__(self, test_cfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)
        self.data_shim = get_data_shim(encoder)
        self.benchmarker = Benchmarker()
        self.test_metric_rows = []
    def _encode_with_cameras(
        self,
        context: dict,
        **kwargs,
    ) -> tuple[object, PredictedCameras | None]:
        return self.encoder.forward_with_cameras(context, **kwargs)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return run_test_step(self, batch, batch_idx)

    def test_step_align(
        self,
        batch,
        gaussians,
        initial_extrinsics,
        intrinsics,
        near,
        far,
        verbose=False,
    ):
        return run_test_step_align(
            self, batch, gaussians, initial_extrinsics, intrinsics, near, far,
            verbose=verbose,
        )

    def on_test_start(self) -> None:
        self.test_metric_rows = []

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        metric_path = self.test_cfg.output_path / name / "metrics.json"
        metric_path.parent.mkdir(exist_ok=True, parents=True)
        with metric_path.open("w") as f:
            json.dump(self.test_metric_rows, f, indent=2)
        budget_auc_rows = [
            row["budget_auc"]
            for row in self.test_metric_rows
            if "budget_auc" in row
        ]
        if budget_auc_rows:
            budget_auc_summary = {
                metric_name: sum(
                    row[metric_name] for row in budget_auc_rows
                ) / len(budget_auc_rows)
                for metric_name in budget_auc_rows[0]
            }
            budget_auc_path = metric_path.parent / "budget_auc.json"
            with budget_auc_path.open("w") as f:
                json.dump(
                    {
                        "num_scenes": len(budget_auc_rows),
                        "normalized_auc": budget_auc_summary,
                    },
                    f,
                    indent=2,
                )
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.dump_tagged_memory(
            self.test_cfg.output_path / name / "peak_memory_by_tag.json"
        )
        self.benchmarker.summarize()

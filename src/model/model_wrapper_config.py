from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    gaussian_budget_per_batch: int | None = None
    gaussian_budget_sweep: bool = False
    gaussian_budget_per_view_list: list[int] | None = None

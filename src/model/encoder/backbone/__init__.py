from typing import Any

import torch.nn as nn

from .backbone import Backbone
from .backbone_dvlt import BackboneDVLT, BackboneDVLTCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "dvlt_multi": BackboneDVLT,
}

BackboneCfg = BackboneDVLTCfg


def get_backbone(
    cfg: BackboneCfg,
    d_in: int = 3,
    predict_cameras: bool = False,
) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in, predict_cameras=predict_cameras)

from dataclasses import fields

from torch.utils.data import Dataset

from .dataset_re10k import (
    DatasetRE10k,
    DatasetRE10kCfg,
    DatasetRE10kCfgWrapper,
)
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
}


DatasetCfgWrapper = DatasetRE10kCfgWrapper
DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
) -> list[Dataset]:
    if stage != "test":
        raise ValueError("This release provides test datasets only.")
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler_cfg = cfg.view_sampler
        view_sampler = get_view_sampler(
            view_sampler_cfg,
            stage,
            cfg.cameras_are_circular,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets

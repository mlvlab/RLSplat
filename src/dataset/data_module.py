"""Fixed-index evaluation dataloaders."""
import random
from dataclasses import dataclass

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader

from . import get_dataset
from .types import DataShim


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    test: DataLoaderStageCfg


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    def __init__(self, dataset_cfgs, data_loader_cfg) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg

    def test_dataloader(self):
        cfg = self.data_loader_cfg.test
        if cfg.batch_size != 1:
            raise ValueError("Evaluation requires batch_size=1.")
        data_loaders = []
        for dataset in get_dataset(self.dataset_cfgs, "test"):
            generator = None
            if cfg.seed is not None:
                generator = torch.Generator().manual_seed(cfg.seed)
            data_loaders.append(DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                generator=generator,
                worker_init_fn=worker_init_fn,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            ))
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

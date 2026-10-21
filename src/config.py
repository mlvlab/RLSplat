from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper_config import TestCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["test"]
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers,
         list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers},
    )

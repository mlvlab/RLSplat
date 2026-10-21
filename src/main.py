"""Evaluation entrypoint for released RLSplat checkpoints."""
from pathlib import Path

import hydra
from omegaconf import DictConfig



def validate_evaluation_config(cfg: DictConfig) -> None:
    if cfg.mode != "test":
        raise ValueError("This release supports evaluation only; use mode=test.")
    if not cfg.checkpointing.load:
        raise ValueError("Evaluation requires checkpointing.load=<checkpoint.ckpt>.")
    checkpoint_ref = str(cfg.checkpointing.load)
    if not checkpoint_ref.startswith("hf://") and not Path(checkpoint_ref).is_file():
        raise FileNotFoundError(cfg.checkpointing.load)
    if cfg.data_loader.test.batch_size != 1:
        raise ValueError("Evaluation requires data_loader.test.batch_size=1.")
    if cfg.wandb.mode != "disabled":
        raise ValueError("This release uses local outputs; set wandb.mode=disabled.")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def evaluate(cfg_dict: DictConfig) -> None:
    validate_evaluation_config(cfg_dict)

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger

    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.evaluation.checkpoint import load_evaluation_checkpoint
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    seed_everything(cfg.seed, workers=True)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    encoder, _ = get_encoder(cfg.model.encoder)
    model = ModelWrapper(
        cfg.test,
        encoder,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss) if cfg.test.align_pose else [],
    )
    load_evaluation_checkpoint(model, cfg.checkpointing.load)
    model.requires_grad_(False)
    model.eval()
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=CSVLogger(output_dir, name="metrics"),
        enable_checkpointing=False,
        inference_mode=not cfg.test.align_pose,
        use_distributed_sampler=False,
    )
    torch.cuda.empty_cache()
    trainer.test(model, datamodule=DataModule(cfg.dataset, cfg.data_loader))


if __name__ == "__main__":
    evaluate()

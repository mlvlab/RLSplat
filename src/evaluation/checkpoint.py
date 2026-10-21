"""Load inference weights without restoring training state."""
from pathlib import Path

import torch
from torch import nn

from ..misc.legacy_names import (
    remap_legacy_state_dict_keys,
)


TRAINING_ONLY_PREFIXES = (
    "encoder.gaussian_allocation_logit_head_ref.",
    "encoder.gaussian_allocation_budget_film_ref.",
    "losses.",
)
TRAINING_ONLY_KEYS = {"encoder.gaussian_allocation_ref_synced"}


def resolve_hub_checkpoint(reference: str | Path) -> Path:
    """Resolve a local checkpoint path or ``hf://owner/repo/path`` reference."""
    value = str(reference)
    if not value.startswith("hf://"):
        return Path(value)
    hub_path = value[5:]
    parts = hub_path.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Hugging Face checkpoint references must be "
            "hf://<owner>/<repo>/<path-in-repository>."
        )
    from huggingface_hub import hf_hub_download

    repo_id = "/".join(parts[:2])
    return Path(hf_hub_download(repo_id=repo_id, filename=parts[2], repo_type="model"))


PUBLIC_ENCODER_PREFIXES = (
    ("policy_head.", "gaussian_allocation_logit_head."),
    ("policy_budget_film.", "gaussian_allocation_budget_film."),
    ("local_refinement.local_cross_attentions.", "position_topk_cross_transformer.geometry_sparse_attentions."),
    ("local_refinement.", "position_topk_cross_transformer."),
    ("initial_gaussian_head.", "gaussian_parameter_head."),
    ("residual_head.", "gaussian_parameter_residual_head."),
    ("gaussian_token_generator.slot_pe_mlp.", "gaussian_token_generator.slot_fourier_mlp."),
)


def remap_public_checkpoint_keys(state_dict: dict) -> dict:
    """Translate the archival export's encoder module names without changing tensors."""
    result = {}
    for key, value in state_dict.items():
        mapped_key = key
        for public, internal in PUBLIC_ENCODER_PREFIXES:
            prefix = f"encoder.{public}"
            if key.startswith(prefix):
                mapped_key = f"encoder.{internal}" + key[len(prefix):]
                break
        if mapped_key in result:
            raise ValueError(f"Checkpoint keys collide after name mapping: {mapped_key}")
        result[mapped_key] = value
    return result


def load_evaluation_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = resolve_hub_checkpoint(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Expected a Lightning checkpoint containing 'state_dict'.")
        state_dict = checkpoint["state_dict"]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key not in TRAINING_ONLY_KEYS
        and not key.startswith(TRAINING_ONLY_PREFIXES)
    }
    model_keys = set(model.state_dict())
    if not set(state_dict).issubset(model_keys):
        public_state_dict = remap_public_checkpoint_keys(state_dict)
        if set(public_state_dict).issubset(model_keys):
            state_dict = public_state_dict
        else:
            state_dict = remap_legacy_state_dict_keys(
                state_dict, label="evaluation"
            )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if key not in TRAINING_ONLY_KEYS
        and not key.startswith(TRAINING_ONLY_PREFIXES)
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the evaluation model: "
            f"missing={missing[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )

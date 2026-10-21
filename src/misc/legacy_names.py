from __future__ import annotations

import re

STATE_DICT_KEY_RULES: list[tuple[str, str]] = [
    (r"(^|\.)child_generator\.", r"\1gaussian_token_generator."),
    (r"(^|\.)child_allocation_", r"\1gaussian_allocation_"),
    (r"(^|\.)child_to_gaussian_parameter_residual\.", r"\1gaussian_parameter_residual_head."),
    (r"(^|\.)child_to_gaussian_parameters\.", r"\1gaussian_parameter_head."),
    (r"\.parent_film\.", ".anchor_film."),
    (r"\.parent_allocation_encoder\.", ".anchor_allocation_encoder."),
    (r"\.parent_cross_attn\.", ".anchor_cross_attn."),
    (r"\.child_norm\.", ".gaussian_norm."),
    (r"\.to_child_residual\.", ".to_gaussian_residual."),
    (r"(^|\.)gaussian_allocation_logit_head\.", r"\1policy_head."),
    (r"(^|\.)gaussian_allocation_budget_film\.", r"\1policy_budget_film."),
    (r"(^|\.)gaussian_parameter_residual_head\.", r"\1residual_head."),
    (r"(^|\.)gaussian_parameter_head\.", r"\1initial_gaussian_head."),
    (r"(^|\.)position_topk_cross_transformer\.", r"\1local_refinement."),
    (r"\.geometry_sparse_attentions\.", ".local_cross_attentions."),
    (r"(^|\.)gaussian_token_generator\.slot_fourier_mlp\.", r"\1gaussian_token_generator.slot_pe_mlp."),
]
_STATE_DICT_KEY_RULES = [(re.compile(p), r) for p, r in STATE_DICT_KEY_RULES]

REMOVED_STATE_DICT_PREFIXES: tuple[str, ...] = (
    "child_marginal_utility_head.",
    "child_to_position.",
    "child_to_position_residual.",
    "child_to_attributes.",
    "child_to_attribute_residual.",
    "gaussian_allocation_logit_head_ref.",
    "gaussian_allocation_budget_film_ref.",
    "gaussian_allocation_ref_synced",
    "gaussian_token_generator.self_attention.",
)


def rename_state_dict_key(key: str) -> str | None:
    bare = key[len("encoder."):] if key.startswith("encoder.") else key
    if bare.startswith(REMOVED_STATE_DICT_PREFIXES):
        return None
    for pattern, replacement in _STATE_DICT_KEY_RULES:
        key = pattern.sub(replacement, key)
    bare = key[len("encoder."):] if key.startswith("encoder.") else key
    if bare.startswith(REMOVED_STATE_DICT_PREFIXES):
        return None
    return key


def remap_legacy_state_dict_keys(state_dict: dict, *, notice=print, label: str = "") -> dict:
    renamed = 0
    dropped: list[str] = []
    out = {}
    for key, value in state_dict.items():
        new_key = rename_state_dict_key(key)
        if new_key is None:
            dropped.append(key)
            continue
        if new_key != key:
            renamed += 1
        out[new_key] = value
    if renamed == 0 and not dropped:
        return state_dict
    notice(
        f"[legacy checkpoint{(' ' + label) if label else ''}] renamed {renamed} "
        f"state_dict keys to the current vocabulary"
        + (f", dropped {len(dropped)} keys of deleted modules (e.g. {dropped[0]})" if dropped else "")
    )
    return out

"""Gaussian-budget sweeps for validation and test: which budgets to score a
scene at, and the area-under-curve summary of a metric across them."""
import torch
from torch import Tensor


def get_num_gaussians_for_log(gaussians) -> Tensor:
    if isinstance(gaussians, list):
        if not gaussians:
            return torch.tensor(0.0)
        counts = [g.means.new_tensor(g.means.shape[0], dtype=torch.float32) for g in gaussians]
        return torch.stack(counts).mean()
    return gaussians.means.new_tensor(gaussians.means.shape[1], dtype=torch.float32)


def select_validation_budgets(
    min_budget: int,
    max_budget: int,
    num_anchor_tokens: int,
    max_samples: int = 5,
) -> list[int]:
    if min_budget > max_budget:
        raise ValueError("min_budget must not exceed max_budget")
    if num_anchor_tokens < 1:
        raise ValueError("num_anchor_tokens must be positive")
    if max_samples < 2:
        raise ValueError("max_samples must be at least two")

    first_multiplier = (
        min_budget + num_anchor_tokens - 1
    ) // num_anchor_tokens
    last_multiplier = max_budget // num_anchor_tokens
    candidates = {min_budget, max_budget}
    candidates.update(
        multiplier * num_anchor_tokens
        for multiplier in range(first_multiplier, last_multiplier + 1)
    )
    candidates = sorted(
        budget
        for budget in candidates
        if min_budget <= budget <= max_budget
    )
    if len(candidates) <= max_samples:
        return candidates

    last_idx = len(candidates) - 1
    sampled_indices = [
        (sample_idx * last_idx + (max_samples - 1) // 2)
        // (max_samples - 1)
        for sample_idx in range(max_samples)
    ]
    return [candidates[idx] for idx in sampled_indices]


def normalized_budget_auc(
    budgets: list[int],
    metric_values: list[Tensor],
) -> Tensor:
    if len(budgets) != len(metric_values) or not budgets:
        raise ValueError("budgets and metric_values must have equal non-zero length")
    if len(budgets) == 1:
        return metric_values[0]
    budget_tensor = metric_values[0].new_tensor(budgets, dtype=torch.float32)
    normalized_budget = (
        budget_tensor - budget_tensor[0]
    ) / (budget_tensor[-1] - budget_tensor[0]).clamp_min(1.0)
    values = torch.stack([value.float() for value in metric_values])
    return torch.trapz(values, normalized_budget)


def supports_validation_budget_sweep(encoder) -> bool:
    return hasattr(encoder, "_resolve_gaussian_budget_range")


def resolve_test_budgets(
    encoder,
    num_anchor_tokens: int,
    gaussian_budget_per_batch: int | None,
    gaussian_budget_sweep: bool,
    num_views: int = 1,
    per_view_list: list[int] | None = None,
) -> tuple[list[int | None], int | None]:
    if not gaussian_budget_sweep:
        return [gaussian_budget_per_batch], gaussian_budget_per_batch
    if not supports_validation_budget_sweep(encoder):
        raise ValueError(
            "test.gaussian_budget_sweep=true requires an encoder with a "
            "configurable gaussian-budget range"
        )

    if per_view_list:
        budgets = sorted({int(rung) * num_views for rung in per_view_list})
        if gaussian_budget_per_batch is None:
            primary_budget = budgets[len(budgets) // 2]
        else:
            primary_budget = gaussian_budget_per_batch
            if primary_budget not in budgets:
                budgets = sorted([*budgets, primary_budget])
        return budgets, primary_budget

    min_budget, max_budget = encoder._resolve_gaussian_budget_range(
        num_anchor_tokens, num_views
    )
    budgets = select_validation_budgets(
        min_budget,
        max_budget,
        num_anchor_tokens,
    )
    if gaussian_budget_per_batch is None:
        midpoint_budget = (min_budget + max_budget) // 2
        primary_budget = min(
            budgets,
            key=lambda budget: abs(budget - midpoint_budget),
        )
    else:
        if not min_budget <= gaussian_budget_per_batch <= max_budget:
            raise ValueError(
                "test.gaussian_budget_per_batch must lie inside the encoder "
                f"budget range [{min_budget}, {max_budget}], got "
                f"{gaussian_budget_per_batch}"
            )
        primary_budget = gaussian_budget_per_batch
        if primary_budget not in budgets:
            budgets = sorted([*budgets, primary_budget])
    return budgets, primary_budget

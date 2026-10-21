"""Largest-remainder allocation and capacity projection."""
import torch
from torch import Tensor


@torch.no_grad()
def largest_remainder_allocation(
    probs: Tensor,
    batch_budgets: Tensor,
    jitter: bool = False,
) -> Tensor:
    batch_size, num_tokens = probs.shape
    budgets = batch_budgets.long()
    target = probs.float() * budgets.unsqueeze(-1).float()
    counts = target.floor().long()
    deficit = budgets - counts.sum(dim=-1)
    fractional = target - counts.float()
    scatter = torch.argsort(
        (
            torch.arange(num_tokens, device=probs.device, dtype=torch.long)
            * 2654435761
        )
        % 2147483647
    )
    if jitter:
        offsets = torch.empty(num_tokens, device=probs.device, dtype=fractional.dtype)
        offsets[scatter] = (
            num_tokens - 1
            - torch.arange(num_tokens, device=probs.device, dtype=fractional.dtype)
        ) / num_tokens
        fractional = fractional + offsets.unsqueeze(0)
    permuted = torch.argsort(
        fractional[:, scatter], dim=-1, descending=True, stable=True
    )
    order = scatter[permuted]
    rank = torch.arange(num_tokens, device=probs.device).unsqueeze(0)
    gets_extra = rank < deficit.clamp_min(0).unsqueeze(-1)
    counts.scatter_add_(
        dim=1,
        index=order,
        src=gets_extra.long(),
    )
    return counts


@torch.no_grad()
def project_counts_onto_capacity(
    counts: Tensor,
    max_gaussians: int,
) -> Tensor:
    capped = counts.clamp(max=max_gaussians)
    overflow = (counts - capped).sum(dim=-1)
    if not bool((overflow > 0).any()):
        return capped
    batch_size, num_tokens = counts.shape
    order = torch.argsort(capped, dim=-1, descending=True, stable=True)
    headroom = (max_gaussians - capped).gather(-1, order)
    prev_fill = headroom.cumsum(dim=-1) - headroom
    take = torch.minimum(
        headroom,
        (overflow.unsqueeze(-1) - prev_fill).clamp_min(0),
    )
    if bool((take.sum(dim=-1) < overflow).any()):
        worst = int(torch.argmax(overflow - take.sum(dim=-1)))
        raise ValueError(
            "Gaussian budget exceeds what the token grid can hold: "
            f"{int(counts[worst].sum())} gaussians over "
            f"{num_tokens} anchors at max_gaussians_per_anchor="
            f"{max_gaussians}"
        )
    return capped.scatter_add(-1, order, take)

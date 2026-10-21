import torch
from torch import Tensor, nn


_LOG_RATIO_MIN = 1.0 / 64.0
BUDGET_FEATURE_DIM = 1


def make_gaussian_allocation_logit_head(dim: int, hidden_dim: int) -> nn.Module:
    head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def make_gaussian_allocation_budget_film(
    dim: int,
    hidden_dim: int,
    feature_dim: int = BUDGET_FEATURE_DIM,
) -> nn.Module:
    film = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 2 * dim),
    )
    nn.init.zeros_(film[-1].weight)
    nn.init.zeros_(film[-1].bias)
    return film


def budget_film_features(batch_budgets: Tensor, num_tokens: int) -> Tensor:
    ratio = batch_budgets.float() / float(max(num_tokens, 1))
    return torch.log2(ratio.clamp_min(_LOG_RATIO_MIN)).unsqueeze(-1)

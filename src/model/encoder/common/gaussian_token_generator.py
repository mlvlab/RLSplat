import torch
from torch import Tensor, nn

from .gmae import FeedForward


class GaussianTokenGenerator(nn.Module):
    def __init__(self, dim: int, max_gaussians: int | None = None):
        super().__init__()
        self.max_gaussians = max_gaussians
        self.num_slot_frequencies = 4
        self.register_buffer(
            "slot_frequencies",
            2.0 ** torch.arange(self.num_slot_frequencies),
            persistent=False,
        )
        hidden_dim = min(dim, 256)
        self.slot_fourier_mlp = nn.Sequential(
            nn.Linear(1 + 2 * self.num_slot_frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.anchor_film = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
        )
        nn.init.normal_(self.slot_fourier_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.slot_fourier_mlp[-1].bias)
        nn.init.zeros_(self.anchor_film[-1].weight)
        nn.init.zeros_(self.anchor_film[-1].bias)
        self.ff = FeedForward(dim, dim * 2)

    def _slot_positional_features(self, slot_idx: Tensor, count_idx: Tensor) -> Tensor:
        count = count_idx.float().clamp_min(1.0)
        slot_rel = (slot_idx.float() + 0.5) / count
        frequencies = self.slot_frequencies.to(device=slot_rel.device, dtype=slot_rel.dtype)
        angles = 2.0 * torch.pi * slot_rel.unsqueeze(-1) * frequencies
        bin_std = (1.0 / count).unsqueeze(-1) / (12.0**0.5)
        attenuation = torch.exp(-2.0 * (torch.pi * frequencies * bin_std) ** 2)
        return torch.cat(
            (
                slot_rel.unsqueeze(-1) * 2.0 - 1.0,
                angles.sin() * attenuation,
                angles.cos() * attenuation,
            ),
            dim=-1,
        )

    def forward(self, anchor_tokens: Tensor, gaussian_counts: Tensor) -> tuple[Tensor, Tensor]:
        batch, anchors, dim = anchor_tokens.shape
        counts = gaussian_counts.reshape(batch * anchors)
        if self.max_gaussians is not None:
            counts = counts.clamp(max=self.max_gaussians)
        anchor_idx = torch.repeat_interleave(
            torch.arange(batch * anchors, device=anchor_tokens.device), counts
        )
        slot_idx = torch.arange(anchor_idx.numel(), device=anchor_tokens.device)
        slot_idx = slot_idx - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        anchor_flat = anchor_tokens.reshape(batch * anchors, dim)
        slot = self._slot_positional_features(slot_idx, counts[anchor_idx])
        scale, shift = self.anchor_film(anchor_flat)[anchor_idx].chunk(2, dim=-1)
        embedding = self.slot_fourier_mlp(slot)
        embedding = embedding * (1.0 + torch.tanh(scale)) + shift
        query = anchor_flat[anchor_idx] + embedding
        return query + self.ff(query), anchor_idx

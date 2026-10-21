"""Position-top-k anchor routing and the packed gaussian-token decoder."""
import torch
from torch import Tensor, nn

from .attention import (
    _POSITION_TOPK_MISSING_DEPENDENCIES,
    GeometrySparseGaussianAttention,
    PositionTopKCrossBlock,
    knn_points,
)


class PositionTopKCrossTransformer(nn.Module):
    """Decode gaussian slots from anchor tokens routed by detached 3D-centroid KNN."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        anchor_count: int,
        max_gaussians: int = 1,
        geometry_anchor_count: int = 8,
        geometry_neighbor_count: int = 8,
        anchor_attention: bool = True,
        geometry_attention: bool = True,
        anchor_include_self: bool = True,
        geometry_include_self: bool = False,
        anchor_attention_first: bool = False,
        geometry_ffn: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if anchor_count < 1:
            raise ValueError("anchor_count must be positive")
        if max_gaussians < 1:
            raise ValueError("max_gaussians must be positive")
        if geometry_anchor_count < 1:
            raise ValueError("geometry_anchor_count must be positive")
        if _POSITION_TOPK_MISSING_DEPENDENCIES:
            raise RuntimeError(
                "position_topk_cross requires optimized dependencies: "
                f"{', '.join(_POSITION_TOPK_MISSING_DEPENDENCIES)}. "
                "Fallback execution is disabled."
            )
        self.depth = depth
        self.anchor_count = anchor_count
        self.max_gaussians = max_gaussians
        self.geometry_anchor_count = geometry_anchor_count
        self.anchor_attention_enabled = anchor_attention
        self.geometry_attention_enabled = geometry_attention
        self.anchor_include_self = anchor_include_self
        self.geometry_include_self = geometry_include_self
        self.anchor_attention_first = anchor_attention_first
        with torch.random.fork_rng(devices=[]):
            self.geometry_sparse_attentions = nn.ModuleList(
                [
                    GeometrySparseGaussianAttention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        neighbor_count=geometry_neighbor_count,
                        ffn=geometry_ffn,
                    )
                    for _ in range(depth)
                ]
                if geometry_attention
                else []
            )
        self.blocks = nn.ModuleList(
            [
                PositionTopKCrossBlock(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                )
                for _ in range(depth)
            ]
            if anchor_attention
            else []
        )
        self.gaussian_norm = nn.LayerNorm(dim)

    def _select_anchor_idx_by_anchor(
        self,
        anchor_positions: Tensor,
        anchor_count: int | None = None,
        gaussian_counts: Tensor | None = None,
        include_self: bool = True,
    ) -> Tensor:
        """[batch, anchor, k] anchor indices routed to each anchor.

        include_self=True (the stored behaviour): slot 0 is the anchor
        itself, then its k-1 nearest others. include_self=False: its k
        nearest others, none of them itself (a single anchor still routes
        to itself, there being nothing else).
        """
        batch_size, num_anchors, _ = anchor_positions.shape
        requested_count = self.anchor_count if anchor_count is None else anchor_count
        selected_count = min(requested_count, num_anchors)
        own_idx = torch.arange(num_anchors, device=anchor_positions.device)
        own_idx = own_idx.view(1, num_anchors, 1).expand(
            batch_size,
            -1,
            -1,
        )
        if include_self and selected_count == 1:
            return own_idx
        if not include_self and num_anchors == 1:
            return own_idx
        other_count = selected_count if not include_self else selected_count - 1
        other_count = min(other_count, num_anchors - 1)

        candidate_count = min(num_anchors, selected_count + 1)
        query_positions = anchor_positions.detach().float()
        target_positions = query_positions
        if gaussian_counts is not None:
            has_gaussians = gaussian_counts.reshape(batch_size, num_anchors) > 0
            if not bool(has_gaussians.all()):
                span = (
                    query_positions.amax(dim=1, keepdim=True)
                    - query_positions.amin(dim=1, keepdim=True)
                ).norm(dim=-1, keepdim=True)
                push = span.clamp_min(1.0) * 10.0
                target_positions = torch.where(
                    has_gaussians.unsqueeze(-1),
                    query_positions,
                    query_positions + push,
                )
        candidate_idx = knn_points(
            query_positions,
            target_positions,
            K=candidate_count,
            return_nn=False,
        ).idx
        candidate_rank = torch.arange(
            candidate_count,
            device=anchor_positions.device,
        ).view(1, 1, candidate_count)
        candidate_rank = candidate_rank.masked_fill(
            candidate_idx == own_idx,
            candidate_count,
        )
        keep_idx = candidate_rank.argsort(dim=-1)[..., :other_count]
        neighbor_idx = candidate_idx.gather(-1, keep_idx)
        if not include_self:
            return neighbor_idx
        return torch.cat((own_idx, neighbor_idx), dim=-1)


    def _select_anchor_indices_packed(
        self,
        anchor_positions: Tensor,
        anchor_flat_idx: Tensor,
        anchor_idx_by_anchor: Tensor | None = None,
        gaussian_counts: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_anchors, _ = anchor_positions.shape
        selected_count = min(self.anchor_count, num_anchors)
        if anchor_idx_by_anchor is None:
            anchor_idx_by_anchor = self._select_anchor_idx_by_anchor(
                anchor_positions,
                gaussian_counts=gaussian_counts,
            )
        elif (
            anchor_idx_by_anchor.shape[:2] != anchor_positions.shape[:2]
            or anchor_idx_by_anchor.shape[-1] < selected_count
        ):
            raise ValueError(
                "anchor_idx_by_anchor must cover every anchor and at least "
                f"{selected_count} neighbors, got {tuple(anchor_idx_by_anchor.shape)}"
            )
        anchor_idx_by_anchor = anchor_idx_by_anchor[..., :selected_count]
        batch_idx = torch.div(anchor_flat_idx, num_anchors, rounding_mode="floor")
        own_anchor_idx = anchor_flat_idx - batch_idx * num_anchors
        selected_anchor_idx = anchor_idx_by_anchor[
            batch_idx,
            own_anchor_idx,
        ]
        return (
            batch_idx.unsqueeze(-1) * num_anchors + selected_anchor_idx
        )


    def _select_geometry_gaussian_neighbors_packed(
        self,
        gaussian_positions: Tensor,
        anchor_positions: Tensor,
        anchor_flat_idx: Tensor,
        gaussian_counts: Tensor,
        anchor_idx_by_anchor: Tensor | None,
        neighbor_count: int,
    ) -> tuple[Tensor, Tensor]:
        num_gaussians = gaussian_positions.shape[0]
        if num_gaussians == 0:
            empty_idx = anchor_flat_idx.new_empty((0, 0))
            return empty_idx, empty_idx.bool()

        batch_size, num_anchors = gaussian_counts.shape
        counts_flat = gaussian_counts.reshape(-1).long()
        anchor_starts = counts_flat.cumsum(0) - counts_flat
        gaussian_slot_idx = torch.arange(
            num_gaussians,
            device=anchor_flat_idx.device,
        ) - torch.repeat_interleave(anchor_starts, counts_flat)

        invalid_gaussian_idx = num_gaussians
        max_realized = int(counts_flat.max())
        gaussian_idx_by_anchor = anchor_flat_idx.new_full(
            (batch_size * num_anchors, max(max_realized, 1)),
            invalid_gaussian_idx,
        )
        gaussian_idx_by_anchor[anchor_flat_idx, gaussian_slot_idx] = torch.arange(
            num_gaussians,
            device=anchor_flat_idx.device,
        )

        selected_anchor_count = min(self.geometry_anchor_count, num_anchors)
        if anchor_idx_by_anchor is None:
            anchor_idx_by_anchor = self._select_anchor_idx_by_anchor(
                anchor_positions,
                self.geometry_anchor_count,
                gaussian_counts=gaussian_counts,
            )
        elif (
            anchor_idx_by_anchor.shape[:2] != anchor_positions.shape[:2]
            or anchor_idx_by_anchor.shape[-1] < selected_anchor_count
        ):
            raise ValueError(
                "anchor_idx_by_anchor must cover every anchor and at least "
                f"{selected_anchor_count} neighbors, got "
                f"{tuple(anchor_idx_by_anchor.shape)}"
            )
        anchor_idx_by_anchor = anchor_idx_by_anchor[..., :selected_anchor_count]
        batch_offset = torch.arange(
            batch_size,
            device=anchor_flat_idx.device,
        ).view(batch_size, 1, 1) * num_anchors
        selected_anchor_flat_idx = anchor_idx_by_anchor + batch_offset
        selected_anchor_flat_idx = selected_anchor_flat_idx.view(
            batch_size * num_anchors,
            -1,
        )[anchor_flat_idx]
        candidate_idx = gaussian_idx_by_anchor[selected_anchor_flat_idx].flatten(1)
        candidate_mask = candidate_idx != invalid_gaussian_idx
        safe_candidate_idx = candidate_idx.clamp(max=num_gaussians - 1)

        detached_positions = gaussian_positions.detach().float()
        candidate_positions = detached_positions[safe_candidate_idx]
        distance_squared = (
            candidate_positions - detached_positions.unsqueeze(1)
        ).square().sum(dim=-1)
        own_gaussian_idx = torch.arange(
            num_gaussians,
            device=anchor_flat_idx.device,
        ).unsqueeze(1)
        if not self.geometry_include_self:
            candidate_mask = candidate_mask & (candidate_idx != own_gaussian_idx)
        distance_squared = distance_squared.masked_fill(
            ~candidate_mask,
            torch.inf,
        )

        output_count = min(neighbor_count, selected_anchor_count * self.max_gaussians)
        topk_count = min(output_count, candidate_idx.shape[1])
        selection_cost = distance_squared
        neighbor_cost, neighbor_candidate_idx = selection_cost.topk(
            topk_count,
            dim=-1,
            largest=False,
        )
        neighbor_mask = neighbor_cost.isfinite()
        neighbor_idx = safe_candidate_idx.gather(1, neighbor_candidate_idx)
        neighbor_idx = torch.where(
            neighbor_mask,
            neighbor_idx,
            own_gaussian_idx.expand_as(neighbor_idx),
        )
        if topk_count < output_count:
            pad = output_count - topk_count
            neighbor_idx = torch.cat(
                (neighbor_idx, own_gaussian_idx.expand(num_gaussians, pad)), dim=1
            )
            neighbor_mask = torch.cat(
                (neighbor_mask, neighbor_mask.new_zeros(num_gaussians, pad)), dim=1
            )
        return neighbor_idx, neighbor_mask


    @staticmethod
    def compute_anchor_positions_packed(
        gaussian_positions: Tensor,
        anchor_flat_idx: Tensor,
        gaussian_counts: Tensor,
    ) -> Tensor:
        if gaussian_positions.ndim != 2 or gaussian_positions.shape[-1] != 3:
            raise ValueError(
                "gaussian_positions must have shape [total_gaussians, 3], got "
                f"{tuple(gaussian_positions.shape)}"
            )
        if anchor_flat_idx.shape != (gaussian_positions.shape[0],):
            raise ValueError(
                "anchor_flat_idx and gaussian_positions must have matching packed "
                f"lengths, got {tuple(anchor_flat_idx.shape)} and "
                f"{tuple(gaussian_positions.shape)}"
            )
        if gaussian_counts.ndim != 2:
            raise ValueError(
                "gaussian_counts must have shape [batch, anchor], got "
                f"{tuple(gaussian_counts.shape)}"
            )

        num_anchor_tokens = gaussian_counts.numel()
        anchor_position_sum = gaussian_positions.new_zeros(
            num_anchor_tokens,
            3,
        ).index_add(
            0,
            anchor_flat_idx,
            gaussian_positions,
        )
        anchor_positions = anchor_position_sum / gaussian_counts.reshape(
            num_anchor_tokens,
            1,
        ).clamp_min(1).to(dtype=gaussian_positions.dtype)
        return anchor_positions.view(*gaussian_counts.shape, 3)


    def forward_packed(
        self,
        anchor_tokens: Tensor,
        gaussian_tokens: Tensor,
        gaussian_positions: Tensor,
        anchor_flat_idx: Tensor,
        gaussian_counts: Tensor,
        batch_budgets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if anchor_tokens.shape[:2] != gaussian_counts.shape:
            raise ValueError("anchor token and Gaussian count shapes must match")
        if gaussian_tokens.shape[0] != gaussian_positions.shape[0] or anchor_flat_idx.shape[0] != gaussian_tokens.shape[0]:
            raise ValueError("packed Gaussian tokens, positions, and anchor indices must match")
        if batch_budgets.shape != (anchor_tokens.shape[0],):
            raise ValueError("batch budget count must match the batch size")

        anchor_positions = self.compute_anchor_positions_packed(
            gaussian_positions, anchor_flat_idx, gaussian_counts
        )
        routing_count = max(self.anchor_count, self.geometry_anchor_count)
        anchor_idx_by_anchor = self._select_anchor_idx_by_anchor(
            anchor_positions, routing_count, gaussian_counts=gaussian_counts
        )
        attention_anchor_idx = None
        if self.anchor_attention_enabled:
            if self.anchor_include_self:
                attention_anchor_idx = self._select_anchor_indices_packed(
                    anchor_positions, anchor_flat_idx, anchor_idx_by_anchor
                )
            else:
                attention_anchor_idx = self._select_anchor_indices_packed(
                    anchor_positions,
                    anchor_flat_idx,
                    self._select_anchor_idx_by_anchor(
                        anchor_positions,
                        gaussian_counts=gaussian_counts,
                        include_self=False,
                    ),
                )

        routing = {}
        for layer_idx in range(self.depth):
            stages = ("anchor", "geometry") if self.anchor_attention_first else ("geometry", "anchor")
            for stage in stages:
                if stage == "geometry" and self.geometry_attention_enabled:
                    layer = self.geometry_sparse_attentions[layer_idx]
                    neighbors = routing.get(layer.neighbor_count)
                    if neighbors is None:
                        neighbors = self._select_geometry_gaussian_neighbors_packed(
                            gaussian_positions,
                            anchor_positions,
                            anchor_flat_idx,
                            gaussian_counts,
                            anchor_idx_by_anchor,
                            layer.neighbor_count,
                        )
                        routing[layer.neighbor_count] = neighbors
                    gaussian_tokens, gaussian_positions = layer(
                        gaussian_tokens,
                        gaussian_positions,
                        neighbors[0],
                        neighbors[1],
                        anchor_flat_idx,
                    )
                elif stage == "anchor" and self.anchor_attention_enabled:
                    gaussian_tokens = self.blocks[layer_idx].forward_anchor_cross_indexed_packed(
                        gaussian_tokens,
                        anchor_tokens,
                        attention_anchor_idx,
                        None,
                    )
        return self.gaussian_norm(gaussian_tokens), gaussian_positions

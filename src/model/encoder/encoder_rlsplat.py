from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ..camera_prediction import PredictedCameras
from ..gaussian_allocation.heads import (
    BUDGET_FEATURE_DIM,
    budget_film_features,
    make_gaussian_allocation_budget_film,
    make_gaussian_allocation_logit_head,
)
from ..gaussian_allocation.sampling import (
    largest_remainder_allocation,
    project_counts_onto_capacity,
)
from ..types import Gaussians_b
from .backbone import get_backbone
from .common.anchor_routing import PositionTopKCrossTransformer
from .common.gaussian_adapter import UnifiedGaussianAdapter
from .common.gaussian_token_generator import GaussianTokenGenerator
from .encoder import Encoder
from .encoder_rlsplat_config import EncoderRLSplatCfg

NUM_SURFACES = 1


def make_gaussian_parameter_head(
    dim: int,
    d_sh: int,
    zero_output: bool = False,
) -> nn.Module:
    """Joint MLP head: [position(3), density/scale(4), rotation(4), SH(3*d_sh)]."""
    output_dim = 11 + 3 * d_sh
    head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, 256),
        nn.SiLU(),
        nn.Linear(256, output_dim),
    )
    if zero_output:
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    return head


class EncoderRLSplat(Encoder[EncoderRLSplatCfg]):
    """Predict and refine a budgeted set of Gaussians from context images."""

    backbone: nn.Module
    gaussian_adapter: UnifiedGaussianAdapter

    def __init__(self, cfg: EncoderRLSplatCfg) -> None:
        super().__init__(cfg)

        if cfg.camera_mode not in ("gt_scaled", "predicted"):
            raise ValueError(f"Unknown camera_mode={cfg.camera_mode}")
        self.backbone = get_backbone(
            cfg.backbone,
            3,
            predict_cameras=cfg.camera_mode == "predicted",
        )

        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_size
        backbone_dim = self.backbone.d_out
        transformer_dim = backbone_dim

        if cfg.position_topk_anchor_count < 1:
            raise ValueError("position_topk_anchor_count must be positive")
        if cfg.position_topk_residual_scale < 0.0:
            raise ValueError("position_topk_residual_scale must be non-negative")
        if cfg.position_topk_geometry_anchor_count < 1:
            raise ValueError("position_topk_geometry_anchor_count must be positive")
        if cfg.position_topk_geometry_neighbor_count < 1:
            raise ValueError("position_topk_geometry_neighbor_count must be positive")
        if cfg.max_gaussians_per_anchor < 1:
            raise ValueError("max_gaussians_per_anchor must be positive")
        self.gaussian_allocation_budget_film = None
        hidden_dim = min(transformer_dim, 256)
        self.gaussian_allocation_logit_head = make_gaussian_allocation_logit_head(
            transformer_dim,
            hidden_dim,
        )
        if cfg.gaussian_allocation_budget_conditioning:
            self.gaussian_allocation_budget_film = (
                make_gaussian_allocation_budget_film(
                    transformer_dim,
                    hidden_dim,
                    BUDGET_FEATURE_DIM,
                )
            )
        self.register_buffer(
            "anchor_position",
            torch.tensor([0.0, 0.0, 1.0]),
            persistent=False,
        )

        self.gaussian_token_generator = GaussianTokenGenerator(transformer_dim)
        self.position_topk_cross_transformer = PositionTopKCrossTransformer(
            transformer_dim,
            depth=cfg.decoder_depth,
            heads=16,
            dim_head=transformer_dim // 16,
            anchor_count=cfg.position_topk_anchor_count,
            max_gaussians=cfg.max_gaussians_per_anchor,
            geometry_anchor_count=cfg.position_topk_geometry_anchor_count,
            geometry_neighbor_count=cfg.position_topk_geometry_neighbor_count,
            anchor_attention=cfg.position_topk_anchor_attention,
            geometry_attention=cfg.position_topk_geometry_attention,
            anchor_include_self=cfg.position_topk_anchor_include_self,
            geometry_include_self=cfg.position_topk_geometry_include_self,
            anchor_attention_first=cfg.position_topk_anchor_attention_first,
            geometry_ffn=cfg.position_topk_geometry_ffn,
        )
        self.gaussian_parameter_head = make_gaussian_parameter_head(
            transformer_dim,
            self.gaussian_adapter.d_sh,
        )
        with torch.random.fork_rng(devices=[]):
            self.gaussian_parameter_residual_head = make_gaussian_parameter_head(
                transformer_dim,
                self.gaussian_adapter.d_sh,
                zero_output=True,
            )
    def _compute_gaussian_allocation_logits(self, anchor_tokens: Tensor, batch_budgets: Tensor) -> Tensor:
        hidden = self.gaussian_allocation_logit_head[0](anchor_tokens)
        if self.gaussian_allocation_budget_film is not None:
            features = budget_film_features(batch_budgets, anchor_tokens.shape[1])
            scale, shift = self.gaussian_allocation_budget_film(features).chunk(2, dim=-1)
            hidden = hidden * (1.0 + torch.tanh(scale.unsqueeze(1))) + shift.unsqueeze(1)
        for layer in self.gaussian_allocation_logit_head[1:]:
            hidden = layer(hidden)
        return torch.nan_to_num(hidden.squeeze(-1), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

    @torch.no_grad()
    def _allocate_softmax_proportional_gaussian_counts(
        self,
        logits: Tensor,
        batch_budgets: Tensor,
    ) -> Tensor:
        """Deterministic allocation: largest remainder over B * softmax(l)."""
        probs = torch.softmax(logits.float(), dim=-1)
        counts = largest_remainder_allocation(
            probs,
            batch_budgets,
            jitter=self.cfg.gaussian_allocation_remainder_jitter,
        )
        return project_counts_onto_capacity(
            counts,
            self.cfg.max_gaussians_per_anchor,
        )




    def _resolve_gaussian_budget_range(
        self,
        num_tokens: int,
        num_views: int = 1,
    ) -> tuple[int, int]:
        feasible_min = 0
        feasible_max = num_tokens * self.cfg.max_gaussians_per_anchor
        max_per_view = self.cfg.max_gaussian_budget_per_view
        if self.cfg.min_gaussian_budget_total >= 0:
            min_budget = self.cfg.min_gaussian_budget_total
        else:
            min_budget = feasible_min
        max_budget = (
            max_per_view * num_views if max_per_view >= 0 else feasible_max
        )

        if min_budget > max_budget:
            raise ValueError(
                f"Invalid gaussian budget range: min={min_budget}, max={max_budget}"
            )
        if min_budget < feasible_min or max_budget > feasible_max:
            raise ValueError(
                "Configured gaussian budget range is infeasible for the per-token "
                f"constraints: requested [{min_budget}, {max_budget}], feasible "
                f"[{feasible_min}, {feasible_max}] from "
                f"max_gaussians_per_anchor={self.cfg.max_gaussians_per_anchor}, "
                f"num_tokens={num_tokens}."
            )
        return min_budget, max_budget






    def _allocate_variable_gaussian_counts(
        self,
        context: dict,
        num_tokens: int,
        anchor_tokens: Tensor,
        budget_per_batch_override: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = context["image"].shape[0]
        if batch_size != 1:
            raise ValueError("Evaluation inference requires batch_size=1")
        if budget_per_batch_override is None:
            minimum, maximum = self._resolve_gaussian_budget_range(
                num_tokens,
                context["image"].shape[1],
            )
            budget = (minimum + maximum) // 2
        else:
            budget = budget_per_batch_override
            maximum = num_tokens * self.cfg.max_gaussians_per_anchor
            if not 0 <= budget <= maximum:
                raise ValueError(f"Gaussian budget {budget} is outside [0, {maximum}]")

        budgets = torch.full((batch_size,), budget, device=anchor_tokens.device, dtype=torch.long)
        logits = self._compute_gaussian_allocation_logits(anchor_tokens, budgets)
        counts = self._allocate_softmax_proportional_gaussian_counts(logits, budgets)
        return budgets, counts

    def _predict_initial_gaussian_parameters(
        self,
        initial_gaussian_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raw_gaussian_parameters = self.gaussian_parameter_head(
            initial_gaussian_tokens
        )
        initial_gaussian_positions = (
            raw_gaussian_parameters[..., :3] + self.anchor_position
        )
        initial_gaussian_attributes = raw_gaussian_parameters[..., 3:]
        return initial_gaussian_positions, initial_gaussian_attributes

    def _predict_position_topk_gaussian_residuals(
        self,
        refined_gaussian_tokens: Tensor,
        initial_gaussian_positions: Tensor,
        initial_gaussian_attributes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        deltas = self.gaussian_parameter_residual_head(refined_gaussian_tokens)
        point_deltas = self.cfg.position_topk_residual_scale * deltas[..., :3]
        attribute_deltas = deltas[..., 3:]
        return (
            initial_gaussian_positions + point_deltas,
            initial_gaussian_attributes + attribute_deltas,
        )

    @staticmethod
    def _map_pdf_to_opacity(pdf: Tensor) -> Tensor:
        return 0.5 * (1 - (1 - pdf) + pdf)

    def _build_variable_gaussians(
        self,
        anchor_tokens: Tensor,
        gaussian_counts: Tensor,
        batch_budgets: Tensor,
    ) -> list[Gaussians_b]:
        gaussian_tokens, anchor_flat_idx = self.gaussian_token_generator(
            anchor_tokens,
            gaussian_counts,
        )
        positions, attributes = self._predict_initial_gaussian_parameters(gaussian_tokens)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gaussian_tokens, positions = self.position_topk_cross_transformer.forward_packed(
                anchor_tokens,
                gaussian_tokens,
                positions,
                anchor_flat_idx,
                gaussian_counts,
                batch_budgets,
            )
        positions, attributes = self._predict_position_topk_gaussian_residuals(
            gaussian_tokens,
            positions,
            attributes,
        )
        raw = rearrange(attributes, "n (srf c) -> n srf c", srf=NUM_SURFACES)
        density = raw[..., 0].sigmoid().unsqueeze(-1)
        adapted = self.gaussian_adapter(
            positions.unsqueeze(0).unsqueeze(-2),
            self._map_pdf_to_opacity(density.unsqueeze(0)),
            raw[None, :, :, None, 1:],
        )
        means = adapted.means.reshape(-1, 3)
        covariances = adapted.covariances.reshape(-1, 3, 3)
        harmonics = adapted.harmonics.reshape(-1, 3, adapted.harmonics.shape[-1])
        opacities = adapted.opacities.reshape(-1)
        result = []
        start = 0
        for budget in batch_budgets.tolist():
            end = start + int(budget)
            result.append(
                Gaussians_b(
                    means[start:end],
                    covariances[start:end],
                    harmonics[start:end],
                    opacities[start:end],
                )
            )
            start = end
        return result

    def _forward_with_cameras_impl(
        self,
        context: dict,
        visualization_dump: Optional[dict] = None,
        gaussian_budget_per_batch: int | None = None,
        validation_backbone_cache: dict | None = None,
    ) -> tuple[list[Gaussians_b], PredictedCameras | None]:
        if validation_backbone_cache is not None and self.training:
            raise ValueError("validation_backbone_cache requires evaluation mode")
        image = context["image"]
        cache_key = "encoder_rlsplat_backbone"
        cached = validation_backbone_cache.get(cache_key) if validation_backbone_cache is not None else None
        if cached is None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                decoded, _, patch_start_idx, predicted_cameras = self.backbone(
                    context,
                    return_views=False,
                )
            if validation_backbone_cache is not None:
                validation_backbone_cache[cache_key] = (
                    decoded,
                    patch_start_idx,
                    predicted_cameras,
                )
        else:
            decoded, patch_start_idx, predicted_cameras = cached

        anchor_tokens = rearrange(
            decoded[-1][:, :, patch_start_idx:],
            "b v n d -> b (v n) d",
        )
        num_tokens = anchor_tokens.shape[1]
        budgets, gaussian_counts = self._allocate_variable_gaussian_counts(
            context,
            num_tokens,
            anchor_tokens,
            gaussian_budget_per_batch,
        )
        gaussians = self._build_variable_gaussians(
            anchor_tokens,
            gaussian_counts,
            budgets,
        )
        if visualization_dump is not None:
            visualization_dump["gaussian_counts"] = gaussian_counts.detach()
            visualization_dump["gaussian_budgets"] = budgets.detach()
        return gaussians, predicted_cameras

    def forward_with_cameras(
        self,
        context: dict,
        visualization_dump: Optional[dict] = None,
        gaussian_budget_per_batch: int | None = None,
        validation_backbone_cache: dict | None = None,
    ) -> tuple[list[Gaussians_b], PredictedCameras | None]:
        return self._forward_with_cameras_impl(
            context,
            visualization_dump,
            gaussian_budget_per_batch,
            validation_backbone_cache,
        )

    def forward(
        self,
        context: dict,
        visualization_dump: Optional[dict] = None,
        gaussian_budget_per_batch: int | None = None,
        validation_backbone_cache: dict | None = None,
    ) -> list[Gaussians_b]:
        gaussians, _ = self._forward_with_cameras_impl(
            context,
            visualization_dump,
            gaussian_budget_per_batch,
            validation_backbone_cache,
        )
        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim

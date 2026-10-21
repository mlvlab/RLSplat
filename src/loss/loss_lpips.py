from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS, normalize_tensor
from lpips.lpips import spatial_average, upsample
from torch import Tensor

from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def extract_features(
        self,
        image: Float[Tensor, "batch 3 height width"],
    ) -> tuple[Tensor, ...]:
        """Extract normalized LPIPS features from images in [0, 1]."""
        image = 2 * image - 1
        if self.lpips.version == "0.1":
            image = self.lpips.scaling_layer(image)
        outputs = self.lpips.net.forward(image)
        return tuple(
            normalize_tensor(outputs[layer_idx])
            for layer_idx in range(self.lpips.L)
        )

    def compute_distance(
        self,
        prediction: Float[Tensor, "batch 3 height width"],
        target: Float[Tensor, "target_batch 3 height width"],
        target_features: tuple[Tensor, ...] | None = None,
        target_repeats: int = 1,
    ) -> Tensor:
        """Compute unweighted LPIPS, optionally broadcasting cached target features."""
        if target_repeats < 1:
            raise ValueError(f"target_repeats must be positive, got {target_repeats}")
        if prediction.shape[0] != target_repeats * target.shape[0]:
            raise ValueError(
                "LPIPS prediction and target batches do not match repeats: "
                f"{prediction.shape[0]} vs {target_repeats} * {target.shape[0]}"
            )

        prediction_features = self.extract_features(prediction)
        if target_features is None:
            target_features = self.extract_features(target)
        if len(target_features) != self.lpips.L:
            raise ValueError(
                "LPIPS target feature count does not match network layers: "
                f"{len(target_features)} vs {self.lpips.L}"
            )

        layer_results = []
        target_batch_size = target.shape[0]
        for layer_idx, (prediction_feature, target_feature) in enumerate(
            zip(prediction_features, target_features)
        ):
            if target_feature.shape[0] != target_batch_size:
                raise ValueError(
                    "LPIPS cached target feature batch does not match target: "
                    f"{target_feature.shape[0]} vs {target_batch_size}"
                )
            _, channels, height, width = prediction_feature.shape
            difference = (
                prediction_feature.reshape(
                    target_repeats,
                    target_batch_size,
                    channels,
                    height,
                    width,
                )
                - target_feature.unsqueeze(0)
            ).square().reshape_as(prediction_feature)

            if self.lpips.lpips:
                layer_result = self.lpips.lins[layer_idx](difference)
            else:
                layer_result = difference.sum(dim=1, keepdim=True)
            if self.lpips.spatial:
                layer_result = upsample(
                    layer_result,
                    out_HW=prediction.shape[2:],
                )
            else:
                layer_result = spatial_average(layer_result, keepdim=True)
            layer_results.append(layer_result)

        value = layer_results[0]
        for layer_result in layer_results[1:]:
            value = value + layer_result
        return value

    def forward(
        self,
        prediction: DecoderOutput,
        target_image: Float[Tensor, "batch view 3 height width"],
        target_features: tuple[Tensor, ...] | None = None,
    ) -> Float[Tensor, ""]:
        image = target_image

        if self.cfg.weight == 0:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.compute_distance(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            target_features=target_features,
        )
        return self.cfg.weight * loss.mean()

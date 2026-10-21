import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],
    Float[Tensor, "*#batch 3 3"],
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    images = images[..., :, row : row + h_out, col : col + w_out]

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out
    intrinsics[..., 1, 1] *= h_in / h_out

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],
    Float[Tensor, "*#batch 3 3"],
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(
    example: AnyExample,
    shape: tuple[int, int],
    target_shape: tuple[int, int] | None = None,
) -> AnyExample:
    """Crop images in the example.

    ``target_shape`` lets evaluation score a 252-input model on 256x256 target
    views; when omitted, targets share the context shape as before.
    """
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], target_shape or shape),
    }

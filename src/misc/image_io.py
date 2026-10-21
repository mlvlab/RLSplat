from pathlib import Path
from typing import Union

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor


FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    Image.fromarray(prep_image(image)).save(path)

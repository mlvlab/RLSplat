from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians_b:
    means: Float[Tensor, "gaussian dim"]
    covariances: Float[Tensor, "gaussian dim dim"]
    harmonics: Float[Tensor, "gaussian 3 d_sh"]
    opacities: Float[Tensor, "gaussian"]

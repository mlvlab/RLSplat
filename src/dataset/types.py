from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]




class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]
    intrinsics: Float[Tensor, "batch _ 3 3"]
    image: Float[Tensor, "batch _ _ _ _"]
    near: Float[Tensor, "batch _"]
    far: Float[Tensor, "batch _"]
    index: Int64[Tensor, "batch _"]
    overlap: Float[Tensor, "batch _"]


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews

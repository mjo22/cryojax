from typing import TypeAlias, TypeVar

from jaxtyping import Array, PyTree, Shaped


SearchSolution = TypeVar("SearchSolution")
SearchState = TypeVar("SearchState")
Grid: TypeAlias = PyTree[Shaped[Array, "_ ..."] | None, " Y"]
GridPoint: TypeAlias = PyTree[Shaped[Array, "..."] | None, " Y"]

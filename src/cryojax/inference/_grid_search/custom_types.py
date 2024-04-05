from typing import TypeAlias, TypeVar

from jaxtyping import Array, Int, PyTree, Shaped


SearchSolution = TypeVar("SearchSolution")
SearchState = TypeVar("SearchState")
PyTreeGrid: TypeAlias = PyTree[Shaped[Array, "_ ..."], " Y"]
PyTreeGridPoint: TypeAlias = PyTree[Shaped[Array, "..."], " Y"]
PyTreeGridIndex: TypeAlias = PyTree[Int[Array, "..."], "... Y"]

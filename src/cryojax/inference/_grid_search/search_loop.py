""" """

from collections.abc import Callable
from typing import Any  # , cast

import equinox as eqx
from jaxtyping import Array, PyTree, Shaped

from .custom_types import Out, Y as Y
from .search_method import AbstractGridSearchMethod


@eqx.filter_jit
def run_grid_search(
    fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
    method: AbstractGridSearchMethod,
    grid: PyTree[Shaped[Array, "dim ..."] | None, "Y"],
    args: Any = None,
):
    # fn = eqx.filter_closure_convert(fn, grid, args)  # pyright: ignore
    # fn = cast(Callable[[GridPoint, Any], Out], fn)
    # f_struct = fn.out_struct
    pass

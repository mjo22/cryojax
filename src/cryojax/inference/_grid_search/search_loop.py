"""The main search loop for the grid search."""

import math
from collections.abc import Callable
from typing import Any, Optional  # , cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int

from ...core import tree_take
from .custom_types import Grid, GridPoint
from .search_method import AbstractGridSearchMethod


@eqx.filter_jit
def run_grid_search(
    fn: Callable[[GridPoint, Any], Array],
    method: AbstractGridSearchMethod,
    tree_grid: Grid,
    args: Any = None,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    # fn = eqx.filter_closure_convert(fn, grid, args)
    # fn = cast(Callable[[GridPoint, Any], Out], fn)
    # f_struct = fn.out_struct
    pass


def tree_grid_shape(
    tree_grid: Grid,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> tuple[int, ...]:
    _leading_dim_resolver = jtu.tree_map(
        _LeafLeadingDimension, tree_grid, is_leaf=is_leaf
    )
    _reduce_fn = lambda x, y: (
        x.get() + y.get() if isinstance(x, _LeafLeadingDimension) else x + y.get()
    )
    return jtu.tree_reduce(
        _reduce_fn,
        _leading_dim_resolver,
        is_leaf=lambda x: isinstance(x, _LeafLeadingDimension),
    )


def tree_grid_take(
    tree_grid: Grid,
    flat_tree_grid_index: int | Int[Array, ""] | Int[Array, " _"],
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> GridPoint:
    shape = tree_grid_shape(tree_grid, is_leaf=is_leaf)
    size = math.prod(shape)
    flat_tree_grid_index = eqx.error_if(
        flat_tree_grid_index,
        jnp.logical_or(flat_tree_grid_index < 0, flat_tree_grid_index >= size),
        f"Grid index must be greater than 0 and less than {size}.",
    )
    unraveled_index = jnp.unravel_index(flat_tree_grid_index, shape)
    tree_grid_index = unraveled_index  # TODO
    tree_grid_point = tree_take(
        tree_grid, tree_grid_index, axis=0, mode="promise_in_bounds"
    )
    return tree_grid_point


def _get_leading_dim(array):
    return (jnp.atleast_1d(array).shape[0],)


class _LeafLeadingDimension(eqx.Module):
    _leaf: Any

    def get(self):
        if eqx.is_array(self._leaf):
            return _get_leading_dim(self._leaf)
        else:
            leaves = jtu.tree_leaves(self._leaf)
            if len(leaves) > 0:
                _leading_dim = _get_leading_dim(leaves[0])
                if not all([_get_leading_dim(leaf) == _leading_dim for leaf in leaves]):
                    raise ValueError(
                        "Arrays stored in PyTree leaves should share the same "
                        "leading dimension. Found that this is not true for "
                        f"leaf {self._leaf}."
                    )
                return _leading_dim
            else:
                raise ValueError(f"No arrays found at leaf {self._leaf}")

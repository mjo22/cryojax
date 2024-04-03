"""Utilities used in `cryojax` for PyTree manipulation."""

from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree


def tree_take(
    pytree_of_arrays: PyTree[ArrayLike, " T"],
    pytree_of_indices: PyTree[ArrayLike, "... T"],
    axis: Optional[int] = None,
    mode: Optional[str] = None,
    fill_value: Optional[ArrayLike] = None,
) -> PyTree[Any, " T"]:
    return jtu.tree_map(
        lambda i, l: _leaf_take(i, l, axis=axis, mode=mode, fill_value=fill_value),
        pytree_of_arrays,
        pytree_of_indices,
    )


def _leaf_take(index, leaf, **kwargs):
    _take_fn = lambda array: jnp.take(jnp.atleast_1d(array), index, **kwargs)
    if eqx.is_array(leaf):
        return _take_fn(leaf)
    else:
        return jtu.tree_map(_take_fn, leaf)

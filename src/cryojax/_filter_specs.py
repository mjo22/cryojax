"""
Utilities for creating equinox filter_specs.
"""

from typing import Any, Callable, Optional, Sequence, Union

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree


def get_filter_spec(
    pytree: PyTree,
    where: Callable[[PyTree], Union[Any, Sequence[Any]]],
    *,
    inverse: bool = False,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[bool]:
    if not inverse:
        false_pytree = jtu.tree_map(lambda _: False, pytree)
        return eqx.tree_at(
            where, false_pytree, replace_fn=lambda _: True, is_leaf=is_leaf
        )
    else:
        true_pytree = jtu.tree_map(lambda _: True, pytree)
        return eqx.tree_at(
            where, true_pytree, replace_fn=lambda _: False, is_leaf=is_leaf
        )

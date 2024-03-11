"""
Utilities for creating equinox filtered transformations. These routines
are modified from `zodiax`, which was created for the project dLux: https://louisdesdoigts.github.io/dLux/.
"""

import equinox as eqx
from jaxtyping import PyTree
from functools import wraps, partial
from typing import Callable, Union, Sequence, Any, Optional, Hashable

from ._filter_specs import get_filter_spec


def filter_grad(
    func: Callable,
    where_or_filter_spec: Callable[[PyTree], Union[Any, Sequence[Any]]] | PyTree[bool],
    *,
    inverse: bool = False,
    has_aux: bool = False,
) -> Callable:

    @wraps(func)
    def wrapper(pytree: PyTree, *args: Any, **kwargs: Any):

        if isinstance(where_or_filter_spec, Callable):
            where = where_or_filter_spec
            filter_spec = get_filter_spec(pytree, where, inverse=inverse)
        else:
            filter_spec = where_or_filter_spec

        @partial(eqx.filter_grad, has_aux=has_aux)
        def recombine_fn(pytree_if_true: PyTree, pytree_if_false: PyTree):
            pytree = eqx.combine(pytree_if_true, pytree_if_false)
            return func(pytree, *args, **kwargs)

        pytree_if_true, pytree_if_false = eqx.partition(pytree, filter_spec)
        return recombine_fn(pytree_if_true, pytree_if_false)

    return wrapper


def filter_value_and_grad(
    func: Callable,
    where_or_filter_spec: Callable[[PyTree], Union[Any, Sequence[Any]]] | PyTree[bool],
    *,
    inverse: bool = False,
    has_aux: bool = False,
) -> Callable:

    @wraps(func)
    def wrapper(pytree: PyTree, *args: Any, **kwargs: Any):

        if isinstance(where_or_filter_spec, Callable):
            where = where_or_filter_spec
            filter_spec = get_filter_spec(pytree, where, inverse=inverse)
        else:
            filter_spec = where_or_filter_spec

        @partial(eqx.filter_value_and_grad, has_aux=has_aux)
        def recombine_fn(pytree_if_true: PyTree, pytree_if_false: PyTree):
            pytree = eqx.combine(pytree_if_true, pytree_if_false)
            return func(pytree, *args, **kwargs)

        pytree_if_true, pytree_if_false = eqx.partition(pytree, filter_spec)
        return recombine_fn(pytree_if_true, pytree_if_false)

    return wrapper


def filter_vmap(
    func: Callable,
    where_or_filter_spec: Callable[[PyTree], Union[Any, Sequence[Any]]],
    *,
    inverse: bool = False,
    in_axes: PyTree[Union[int, None, Callable[[Any], Optional[int]]]] = eqx.if_array(
        axis=0
    ),
    out_axes: PyTree[Union[int, None, Callable[[Any], Optional[int]]]] = eqx.if_array(
        axis=0
    ),
    axis_name: Hashable = None,
    axis_size: Optional[int] = None,
) -> Callable:

    @wraps(func)
    def wrapper(pytree: PyTree, *args: Any, **kwargs: Any):

        if isinstance(where_or_filter_spec, Callable):
            where = where_or_filter_spec
            filter_spec = get_filter_spec(pytree, where, inverse=inverse)
        else:
            filter_spec = where_or_filter_spec

        @partial(
            eqx.filter_vmap,
            in_axes=(in_axes, None),
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        def recombine_fn(pytree_if_true: PyTree, pytree_if_false: PyTree):
            pytree = eqx.combine(pytree_if_true, pytree_if_false)
            return func(pytree, *args, **kwargs)

        pytree_if_true, pytree_if_false = eqx.partition(pytree, filter_spec)
        return recombine_fn(pytree_if_true, pytree_if_false)

    return wrapper

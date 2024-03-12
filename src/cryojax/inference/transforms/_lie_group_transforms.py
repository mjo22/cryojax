"""
Implementation of transformations in local tangent spaces of lie groups.
This is a re-implementation of `jaxlie.manifold`.

This was necessary because `cryojax` has its own pattern for `jax`
transformations, using `equinox`. The `jaxlie.manifold.grad` and
`jaxlie.manifold.value_and_grad` are not compatible with the `cryojax`
wrappers for gradient transformations.
"""

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._transforms import AbstractParameterTransform
from ...rotations import AbstractMatrixLieGroup
from ...typing import Real_


def _apply_update_with_tangent_transform(u, p):
    if u is None:
        return p
    elif isinstance(u, LocalTangentTransform):
        matrix_lie_group = type(u.group_element)
        local_tangent = u.transformed_parameter
        return eqx.tree_at(
            lambda p: p.group_element,
            p,
            u.group_element @ matrix_lie_group.exp(local_tangent),
        )
    else:
        return p + u


def _is_none_or_tangent_transform(x):
    return x is None or isinstance(x, LocalTangentTransform)


def apply_updates_with_tangent_transform(model: PyTree, updates: PyTree) -> PyTree:
    """Modifed `eqx.apply_updates` to apply updates to a model
    with `LocalTangentTransform`s.

    This assumes that `updates` are a prefix of `model`.
    """
    return jtu.tree_map(
        _apply_update_with_tangent_transform,
        updates,
        model,
        is_leaf=_is_none_or_tangent_transform,
    )


class LocalTangentTransform(AbstractParameterTransform, strict=True):
    """This class transforms an `AbstractMatrixLieGroup` to its local
    tangent space.

    This class is based on the implementation in `jaxlie.manifold`.

    **Attributes:**

    - `transformed_parameter`: The local tangent vector.
    """

    transformed_parameter: Real_
    group_element: AbstractMatrixLieGroup

    def __init__(self, group_element: AbstractMatrixLieGroup):
        """**Arguments:**

        - `matrix_lie_group`: The matrix lie group to be transformed to its
                              local tangent space.
        """
        self.transformed_parameter = jnp.zeros(
            group_element.tangent_dimension, dtype=float
        )
        self.group_element = group_element

    def get(self):
        """An implementation of the `jaxlie.manifold.rplus`."""
        matrix_lie_group = type(self.group_element)
        return jax.lax.stop_gradient(self.group_element) @ matrix_lie_group.exp(
            self.transformed_parameter
        )

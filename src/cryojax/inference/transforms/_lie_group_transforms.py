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
from jaxtyping import PyTree, Array, Float

from ._transforms import AbstractParameterTransform
from ...rotations import SO3


def _apply_update_with_lie_transform(u, p):
    if u is None:
        return p
    elif isinstance(u, SO3Transform):
        local_tangent = u.transformed_parameter
        return eqx.tree_at(
            lambda p: p.group_element,
            p,
            p.group_element @ SO3.exp(local_tangent),
        )
    else:
        return p + u


def _is_none_or_lie_transform(x):
    return x is None or isinstance(x, SO3Transform)


def apply_updates_with_lie_transform(model: PyTree, updates: PyTree) -> PyTree:
    """Modifed `eqx.apply_updates` to apply updates to a model
    with `LocalTangentTransform`s.

    This assumes that `updates` are a prefix of `model`.
    """
    return jtu.tree_map(
        _apply_update_with_lie_transform,
        updates,
        model,
        is_leaf=_is_none_or_lie_transform,
    )


class SO3Transform(AbstractParameterTransform, strict=True):
    """This class transforms an `AbstractMatrixLieGroup` to its local
    tangent space.

    This class is based on the implementation in `jaxlie.manifold`.

    **Attributes:**

    - `transformed_parameter`: The local tangent vector.
    """

    transformed_parameter: Float[Array, "3"]
    group_element: SO3

    def __init__(self, wxyz: Float[Array, "4"]):
        """**Arguments:**

        - `wxyz`: A quaternion that parameterizes the SO3
                  group element.
        """
        self.transformed_parameter = jnp.zeros(SO3.tangent_dimension, dtype=float)
        self.group_element = SO3(wxyz)

    def get(self) -> Float[Array, "4"]:
        """An implementation of the `jaxlie.manifold.rplus`."""
        return (
            jax.lax.stop_gradient(self.group_element)
            @ SO3.exp(self.transformed_parameter)
        ).wxyz

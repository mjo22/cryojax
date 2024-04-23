"""
Transformations used for reparameterizing cryojax models.
"""

from abc import abstractmethod
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar, field, Module
from jaxtyping import Array, Float, PyTree

from ..._errors import error_if_not_positive, error_if_zero


def _is_transformed(x: Any) -> bool:
    return isinstance(x, AbstractParameterTransform)


def _resolve_transform(x: Any) -> Any:
    if isinstance(x, AbstractParameterTransform):
        return x.get()
    else:
        return x


def resolve_transforms(pytree: PyTree) -> PyTree:
    """Transforms a pytree whose parameters have entries
    that are `AbstractParameterTransform`s back to its
    original parameter space.
    """
    return jtu.tree_map(_resolve_transform, pytree, is_leaf=_is_transformed)


class AbstractParameterTransform(Module, strict=True):
    """Base class for a parameter transformation.

    This interface tries to implement a user-friendly way to get custom
    per-parameter behavior in pytrees, as described in the [equinox docs](
    https://docs.kidger.site/equinox/tricks/#custom-per-parameter-behaviour).

    Typically this is used to do inference in better geometries or to enforce
    parameter constraints, however it can be used generally.

    This is a very simple class. When the class is initialized,
    a parameter is taken to a transformed parameter space. When
    `transform.get()` is called, the parameter is taken back to
    the original parameter space.
    """

    transformed_parameter: AbstractVar[Array]

    @abstractmethod
    def get(self) -> Any:
        """Get the parameter in the original parameter space."""
        raise NotImplementedError


class ExpTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter to its logarithm.

    **Attributes:**

    - `transformed_parameter`: The parameter on a logarithmic scale.
    """

    transformed_parameter: Array

    def __init__(self, parameter: Array):
        """**Arguments:**

        - `parameter`: The parameter to be transformed to a logarithmic
                       scale.
        """
        self.transformed_parameter = jnp.log(error_if_not_positive(parameter))

    def get(self) -> Array:
        """The logarithmic-valued parameter transformed with an exponential."""
        return jnp.exp(self.transformed_parameter)


class RescalingTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter by a scale factor and a shift.

    **Attributes:**

    - `transformed_parameter`: The rescaled parameter.
    """

    transformed_parameter: Array
    scaling: Float[Array, ""] = field(converter=error_if_zero)
    shift: Float[Array, ""]

    def __init__(
        self,
        parameter: Array,
        scaling: Float[Array, ""] | float,
        shift: Float[Array, ""] | float = 0.0,
    ):
        """**Arguments:**

        - `parameter`: The parameter to be rescaled.
        - `scaling`: The scale factor.
        - `shift`: The shift.
        """
        self.scaling = jnp.asarray(scaling)
        self.shift = jnp.asarray(shift)
        self.transformed_parameter = self.scaling * jnp.asarray(parameter) + self.shift

    def get(self) -> Array:
        """The rescaled parameter transformed back to the original scale."""
        return (
            self.transformed_parameter - jax.lax.stop_gradient(self.shift)
        ) / jax.lax.stop_gradient(self.scaling)


class ComposedTransform(AbstractParameterTransform, strict=True):
    """This class composes multiple `AbstractParameterTransforms`.

    **Attributes:**

    - `transformed_parameter`: The transformed parameter.
    - `transforms`: The sequence of `AbstractParameterTransform`s.
    """

    transformed_parameter: Any
    transforms: tuple[AbstractParameterTransform, ...]

    def __init__(
        self,
        parameter: Any,
        transform_fns: Sequence[Callable[[Any], "AbstractParameterTransform"]],
    ):
        """**Arguments:**

        - `parameter`: The parameter to be transformed.
        - `transform_fns`: A sequence of functions that take in
                           a parameter and return an `AbstractParameterTransform`.
        """
        p = jnp.asarray(parameter)
        transforms = []
        for transform_fn in transform_fns:
            transform = transform_fn(p)
            p = transform.transformed_parameter
            transforms.append(transform)
        self.transformed_parameter = p
        self.transforms = tuple(transforms)

    def get(self) -> Any:
        """Transform the `transformed_parameter` back to the original space."""
        parameter = self.transformed_parameter
        transforms = jax.lax.stop_gradient(self.transforms)
        for transform in transforms[::-1]:
            transform = eqx.tree_at(
                lambda t: t.transformed_parameter, transform, parameter
            )
            parameter = transform.get()
        return parameter

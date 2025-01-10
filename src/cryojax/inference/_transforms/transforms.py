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
from jaxtyping import Array, PyTree

from ..._errors import error_if_not_positive


def _is_transformed(x: Any) -> bool:
    return isinstance(x, AbstractPyTreeTransform)


def _resolve_transform(x: Any) -> Any:
    if isinstance(x, AbstractPyTreeTransform):
        return x.get()
    else:
        return x


def resolve_transforms(pytree: PyTree) -> PyTree:
    """Transforms a pytree whose parameters have entries
    that are `AbstractParameterTransform`s back to its
    original parameter space.
    """
    return jtu.tree_map(_resolve_transform, pytree, is_leaf=_is_transformed)


class AbstractPyTreeTransform(Module, strict=True):
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

    transformed_pytree: AbstractVar[PyTree[Array]]

    @abstractmethod
    def get(self) -> Any:
        """Get the parameter in the original parameter space."""
        raise NotImplementedError


class CustomTransform(AbstractPyTreeTransform, strict=True):
    """This class transforms a pytree of arrays using a custom callable.

    **Attributes:**

    - `transformed_pytree`: The transformed pytree of arrays.
    """

    transformed_pytree: PyTree[Array]
    fn: Callable[..., Array] = field(static=True)
    args: tuple[Any, ...] = field(static=True)
    kwargs: dict[str, Any] = field(static=True)

    def __init__(
        self,
        fn: Callable[..., Array],
        transformed_pytree: PyTree[Array],
        *args: Any,
        **kwargs: Any,
    ):
        """**Arguments:**

        - `fn`:
            The function to apply to each pytree leaf of `transformed_pytree`.
            This should be of format `fn(leaf, *args, **kwargs)`.
        - `transformed_pytree`:
            The pytree of arrays, already in the transformed space.
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.transformed_pytree = transformed_pytree

    def get(self) -> PyTree[Array]:
        """The pytree transformed with the custom function `fn`."""
        return jax.tree.map(
            lambda leaf: self.fn(leaf, *self.args, **self.kwargs), self.transformed_pytree
        )


class LogTransform(AbstractPyTreeTransform, strict=True):
    """This class transforms a pytree of arrays to their logarithm.

    **Attributes:**

    - `transformed_pytree`: The pytree of parameters on a logarithmic scale.
    """

    transformed_pytree: PyTree[Array]

    def __init__(self, pytree: PyTree[Array]):
        """**Arguments:**

        - `pytree`: The pytree of arrays to be transformed to a logarithmic
                    scale.
        """
        self.transformed_pytree = jax.tree.map(
            lambda array: jnp.log(error_if_not_positive(array)), pytree
        )

    def get(self) -> PyTree[Array]:
        """The logarithmic-valued pytree of arrays transformed with an exponential."""
        return jax.tree.map(lambda array: jnp.exp(array), self.transformed_pytree)


class RescalingTransform(AbstractPyTreeTransform, strict=True):
    """This class transforms a pytree of arrays by a scale factor and a shift.

    **Attributes:**

    - `transformed_pytree`: The rescaled pytree of arrays.
    """

    transformed_pytree: PyTree[Array]
    scaling: float = field(static=True)
    shift: float = field(static=True)

    def __init__(
        self,
        pytree: PyTree[Array],
        scaling: float,
        shift: float = 0.0,
    ):
        """**Arguments:**

        - `pytree`:
            The pytree of arrays to be rescaled.
        - `scaling`:
            The scale factor. This should have the same units of the arrays in `pytree`.
        - `shift`:
            The shift. This should have the same units of the
            arrays in `transformed_pytree`.
        """
        self.scaling = scaling
        self.shift = shift
        self.transformed_pytree = jax.tree.map(
            lambda array: jnp.asarray(array) / self.scaling + self.shift, pytree
        )

    def get(self) -> PyTree[Array]:
        """The pytree of arrays transformed back to the original scale."""
        return jax.tree.map(
            lambda array: (array - self.shift) * self.scaling, self.transformed_pytree
        )


class ComposedTransform(AbstractPyTreeTransform, strict=True):
    """This class composes multiple `AbstractPyTreeTransforms`.

    **Attributes:**

    - `transformed_pytree`: The transformed pytree of arrays.
    - `transforms`: The sequence of `AbstractPyTreeTransform`s.
    """

    transformed_pytree: Any
    transforms: tuple[AbstractPyTreeTransform, ...]

    def __init__(
        self,
        pytree: Any,
        transform_fns: Sequence[Callable[[Any], "AbstractPyTreeTransform"]],
    ):
        """**Arguments:**

        - `pytrees`: The pytree of arrays to be transformed.
        - `transform_fns`: A sequence of functions that take in
                           a pytree and return an `AbstractPyTreeTransform`.
        """
        p = jnp.asarray(pytree)
        transforms = []
        for transform_fn in transform_fns:
            transform = transform_fn(p)
            p = transform.transformed_pytree
            transforms.append(transform)
        self.transformed_pytree = p
        self.transforms = tuple(transforms)

    def get(self) -> Any:
        """Transform the `transformed_pytree` back to the original space."""
        p = self.transformed_pytree
        transforms = jax.lax.stop_gradient(self.transforms)
        for transform in transforms[::-1]:
            transform = eqx.tree_at(lambda t: t.transformed_pytree, transform, p)
            p = transform.get()
        return p

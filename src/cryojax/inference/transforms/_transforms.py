"""
Transformations used for reparameterizing cryojax models.
"""

from abc import abstractmethod
from equinox import Module, AbstractVar, field
from jaxtyping import PyTree, Array, ArrayLike
from typing import Callable, Union, Any, Sequence, Optional
from typing_extensions import overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ...typing import Real_
from ...core import error_if_zero, error_if_not_positive


def _is_transformed(x: Any) -> bool:
    return isinstance(x, AbstractParameterTransform)


def _resolve_transform(x: Any) -> Any:
    if isinstance(x, AbstractParameterTransform):
        return x.get()
    else:
        return x


def _apply_transform(
    pytree: PyTree,
    where: Callable[[PyTree], Union[Any, Sequence[Any]]],
    replace_fn: Callable[[Any], "AbstractParameterTransform"],
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    return eqx.tree_at(where, pytree, replace_fn=replace_fn, is_leaf=is_leaf)


@overload
def insert_transforms(
    pytree: PyTree,
    wheres: Sequence[Callable[[PyTree], Union[Any, Sequence[Any]]]],
    replace_fns: Sequence[Callable[[Any], "AbstractParameterTransform"]],
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree: ...


@overload
def insert_transforms(
    pytree: PyTree,
    wheres: Callable[[PyTree], Union[Any, Sequence[Any]]],
    replace_fns: Callable[[Any], "AbstractParameterTransform"],
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree: ...


def insert_transforms(
    pytree: PyTree,
    wheres: (
        Callable[[PyTree], Union[Any, Sequence[Any]]]
        | Sequence[Callable[[PyTree], Union[Any, Sequence[Any]]]]
    ),
    replace_fns: (
        Callable[[Any], "AbstractParameterTransform"]
        | Sequence[Callable[[Any], "AbstractParameterTransform"]]
    ),
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """Applies an `AbstractParameterTransform` to pytree node(s).

    This function performs a sequence of `equinox.tree_at` calls to apply each `replace_fn`
    in `replace_fns` to each `where` in `wheres`.
    """
    if isinstance(replace_fns, Callable) and isinstance(wheres, Callable):
        where, replace_fn = wheres, replace_fns
        return _apply_transform(pytree, where, replace_fn, is_leaf=is_leaf)
    elif isinstance(replace_fns, Sequence) and isinstance(wheres, Sequence):
        if len(replace_fns) != len(wheres):
            raise TypeError(
                "If arguments `wheres` and `replace_fns` are sequences, they "
                f"must be sequences of the same length. Got `wheres, replace_fns = {wheres}, {replace_fns}`."
            )
        transformed_pytree = pytree
        for where, replace_fn in zip(wheres, replace_fns):
            transformed_pytree = _apply_transform(
                pytree, where, replace_fn, is_leaf=is_leaf
            )
        return transformed_pytree
    else:
        raise TypeError(
            "Input arguments `wheres` and `replace_fns` must both either be functions "
            f"or sequences. Got `wheres, replace_fns = {wheres}, {replace_fns}`."
        )


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
    def get(self):
        """Get the parameter in the original parameter space."""
        raise NotImplementedError


class LogTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter to its logarithm.

    **Attributes:**

    - `transformed_parameter`: The parameter on a logarithmic scale.
    """

    transformed_parameter: Real_

    def __init__(self, parameter: Real_):
        """**Arguments:**

        - `parameter`: The parameter to be transformed to a logarithmic
                       scale.
        """
        self.transformed_parameter = jnp.log(error_if_not_positive(parameter))

    def get(self):
        """The logarithmic-valued parameter transformed with an exponential."""
        return jnp.exp(self.transformed_parameter)


class RescalingTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter by a scale factor and a shift.

    **Attributes:**

    - `transformed_parameter`: The rescaled parameter.
    """

    transformed_parameter: Real_
    scaling: Real_ = field(converter=error_if_zero)
    shift: Real_

    def __init__(
        self,
        parameter: Real_ | float,
        scaling: Real_ | float,
        shift: Real_ | float = 0.0,
    ):
        """**Arguments:**

        - `parameter`: The parameter to be rescaled.

        - `scaling`: The scale factor.

        - `shift`: The shift.
        """
        self.scaling = jnp.asarray(scaling)
        self.shift = jnp.asarray(shift)
        self.transformed_parameter = self.scaling * jnp.asarray(parameter) + self.shift

    def get(self):
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

    transformed_parameter: Array
    transforms: tuple[AbstractParameterTransform, ...]

    def __init__(
        self,
        parameter: ArrayLike,
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

    def get(self):
        """Transform the `transformed_parameter` back to the original space."""
        parameter = self.transformed_parameter
        transforms = jax.lax.stop_gradient(self.transforms)
        for transform in transforms[::-1]:
            transform = eqx.tree_at(
                lambda t: t.transformed_parameter, transform, parameter
            )
            parameter = transform.get()
        return parameter

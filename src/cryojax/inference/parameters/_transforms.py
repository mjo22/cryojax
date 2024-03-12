"""
Transformations used for reparameterizing cryojax models.
"""

from abc import abstractmethod
from equinox import Module, field
from jaxtyping import PyTree
from typing import Callable, Union, Any, Sequence, Optional
from typing_extensions import overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ...typing import Real_
from ...core import error_if_zero


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


@overload
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
        return _apply_transform(pytree, where, replace_fn)
    elif isinstance(replace_fns, Sequence) and isinstance(wheres, Sequence):
        if len(replace_fns) != len(wheres):
            raise TypeError(
                "If arguments `wheres` and `replace_fns` are sequences, they "
                f"must be sequences of the same length. Got `wheres, replace_fns = {wheres}, {replace_fns}`."
            )
        transformed_pytree = pytree
        for where, replace_fn in zip(wheres, replace_fns):
            transformed_pytree = _apply_transform(pytree, where, replace_fn)
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

    @abstractmethod
    def get(self):
        """Get the parameter in the original parameter space."""
        raise NotImplementedError


class ExpTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter to its logarithm.

    **Attributes:**

    - `log_parameter`: The parameter on a logarithmic scale.
    """

    log_parameter: Real_

    def __init__(self, parameter: Real_):
        """**Arguments:**

        - `parameter`: The parameter to be transformed to a logarithmic
                       scale.
        """
        self.log_parameter = jnp.log(parameter)

    def get(self):
        """The `log_parameter` transformed back with an exponential."""
        return jnp.exp(self.log_parameter)


class RescalingTransform(AbstractParameterTransform, strict=True):
    """This class transforms a parameter by a scale factor.

    **Attributes:**

    - `rescaled_parameter`: The rescaled parameter.
    """

    rescaled_parameter: Real_
    rescaling: Real_ = field(converter=error_if_zero)

    def __init__(self, parameter: Real_, rescaling: float):
        """**Arguments:**

        - `parameter`: The parameter to be rescaled.

        - `rescaling`: The scale factor.
        """
        self.rescaled_parameter = jax.lax.stop_gradient(rescaling) * parameter
        self.rescaling = rescaling

    def get(self):
        """The rescaled `rescaled_parameter` transformed back to the original scale."""
        return self.rescaled_parameter / jax.lax.stop_gradient(self.rescaling)

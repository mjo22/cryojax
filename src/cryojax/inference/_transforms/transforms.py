"""
Transformations used for reparameterizing cryojax models.
"""

from abc import abstractmethod
from types import FunctionType
from typing import Any, Callable, Generic, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
from equinox import field, Module
from jaxtyping import PyTree


T = TypeVar("T")


def _resolve_transforms(pytree, resolve_self: bool):
    def f(leaf):
        if isinstance(leaf, AbstractPyTreeTransform):
            return _resolve_transforms(leaf, resolve_self=False).value
        else:
            return leaf

    def is_leaf(x):
        is_transform = isinstance(x, AbstractPyTreeTransform)
        return is_transform and (resolve_self or x is not pytree)

    return jax.tree.map(f, pytree, is_leaf=is_leaf)


def resolve_transforms(pytree: T) -> T:
    """Transforms a pytree whose parameters have entries
    that are `AbstractParameterTransform`s back to its
    original parameter space.

    If `AbstractParameterTransform`s are nested, the innermost
    nodes are unwrapped first. This function is based on the implementation
    in the package `paramax`.
    """

    return _resolve_transforms(pytree, resolve_self=True)


class AbstractPyTreeTransform(Module, Generic[T], strict=True):
    """Base class for a parameter transformation.

    This interface tries to implement a user-friendly way to get custom
    per-parameter behavior in pytrees, as described in the [equinox docs](
    https://docs.kidger.site/equinox/tricks/#custom-per-parameter-behaviour).

    Typically this is used to do inference in better geometries or to enforce
    parameter constraints, however it can be used generally.

    This is a very simple class. When the class is initialized,
    class fields are stored a transformed parameter space. When
    `transform.value` is called, the fields are taken back to
    the original parameter space.
    """

    @property
    @abstractmethod
    def value(self) -> T:
        """Get the value of the transform."""
        raise NotImplementedError


class CustomTransform(AbstractPyTreeTransform[T], strict=True):
    """This class transforms a pytree of arrays using a custom callable."""

    func_dynamic: Callable[..., T]
    func_static: Callable[..., T] = field(static=True)
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ):
        """**Arguments:**

        - `func`:
            A callable of format `func(*args, **kwargs)`. This may be a
            regular function or an pytree with a `__call__` method, such
            as an `equinox.Module`.
        - `args`:
            Arguments to be passed to `fn` at runtime.
        - `kwargs`:
            Keyword arguments to be passed to `fn` at runtime.
        """
        if isinstance(func, FunctionType):
            func = _Func(func)
        elif not isinstance(func, Callable):
            raise ValueError(
                "Argument `func` must be type `Callable`. If `func` "
                "is an `equinox.Module`, it must have a `__call__` method."
            )
        elif not isinstance(func, PyTree):
            raise ValueError(
                "If argument `func` is not a function, it must be a pytree "
                "with a `__call__` method. For example, this can be achieved"
                " with an `equinox.Module`"
            )
        func_dynamic, func_static = eqx.partition(func, eqx.is_array)
        self.func_dynamic = func_dynamic
        self.func_static = func_static
        self.args = tuple(args)
        self.kwargs = kwargs

    @property
    @override
    def value(self) -> T:
        """The pytree transformed with the custom function `func`."""
        func = eqx.combine(self.func_dynamic, self.func_static)
        return func(*self.args, **self.kwargs)


class StopGradientTransform(AbstractPyTreeTransform[T]):
    """Applies stop gradient to all JAX arrays.

    Based on `NonTrainable` from the package `paramax`.
    """

    pytree_differentiable: T
    pytree_static: T = field(static=True)

    def __init__(self, pytree: T):
        differentiable, static = eqx.partition(pytree, eqx.is_array)
        self.pytree_differentiable = differentiable
        self.pytree_static = static

    @property
    @override
    def value(self) -> T:
        return eqx.combine(
            jax.lax.stop_gradient(self.pytree_differentiable), self.pytree_static
        )


class _Func(eqx.Module):
    _func: FunctionType = field(static=True)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

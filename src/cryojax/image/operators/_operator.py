"""
Base classes for image operators.
"""

from abc import abstractmethod
from typing import Any, Callable
from typing_extensions import override

import jax
import jax.numpy as jnp
from equinox import AbstractVar, field, Module, Partial
from jaxtyping import Array, Float, Inexact


class AbstractImageOperator(Module, strict=True):
    """
    The base class for image operators that contain
    model parameters and compute an ``Array`` at runtime.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        raise NotImplementedError

    def __add__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return SumImageOperator(self, other)
        return SumImageOperator(self, Constant(other))

    def __radd__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return SumImageOperator(other, self)
        return SumImageOperator(Constant(other), self)

    def __sub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return DiffImageOperator(self, other)
        return DiffImageOperator(self, Constant(other))

    def __rsub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return DiffImageOperator(other, self)
        return DiffImageOperator(Constant(other), self)

    def __mul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return ProductImageOperator(self, other)
        return ProductImageOperator(self, Constant(other))

    def __rmul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, (AbstractImageOperator, Partial)):
            return ProductImageOperator(other, self)
        return ProductImageOperator(Constant(other), self)


class AbstractImageMultiplier(Module, strict=True):
    """
    Base class for computing and applying an ``Array`` to an image.

    Attributes
    ----------
    operator :
        The operator. Note that this is automatically
        computed upon instantiation.
    """

    buffer: AbstractVar[
        Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
    ]

    def __call__(
        self, image: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return image * jax.lax.stop_gradient(self.buffer)

    def __mul__(self, other) -> "AbstractImageMultiplier":
        return ProductImageMultiplier(operator1=self, operator2=other)

    def __rmul__(self, other) -> "AbstractImageMultiplier":
        return ProductImageMultiplier(operator1=other, operator2=self)


class Constant(AbstractImageOperator, strict=True):
    """An operator that is a constant."""

    value: Float[Array, "..."] = field(default=1.0, converter=jnp.asarray)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Float[Array, ""]:
        return self.value


Constant.__init__.__doc__ = """**Arguments:**

- `value`: The value of the constant
"""


class Lambda(AbstractImageOperator, strict=True):
    """An operator that calls a custom function."""

    fn: Callable[
        [Array], Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
    ] = field(static=True)

    @override
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.fn(*args, **kwargs)


Lambda.__init__.__doc__ = """**Arguments:**

- `fn`: The `Callable` wrapped into a `AbstractImageOperator`.
"""


class Empirical(AbstractImageOperator, strict=True):
    """This operator stores and returns an array, rather than
    computing one from a model.
    """

    array: (
        Inexact[Array, "... y_dim x_dim"]
        | Inexact[Array, "... z_dim y_dim x_dim"]
        | Float[Array, "..."]
    )
    amplitude: Float[Array, "..."] = field(default=1.0, converter=jnp.asarray)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Inexact[Array, "y_dim x_dim"]:
        """Return the scaled and offset measurement."""
        return self.amplitude * jax.lax.stop_gradient(self.array)


Empirical.__init__.__doc__ = """**Arguments:**

- `array`: The array to be returned upon calling `Empirical`.
- `amplitude`: An amplitude scaling for `array`.
"""


class ProductImageMultiplier(AbstractImageMultiplier, strict=True):
    """A helper to represent the product of two operators."""

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    operator1: AbstractImageMultiplier
    operator2: AbstractImageMultiplier

    def __init__(
        self,
        operator1: AbstractImageMultiplier,
        operator2: AbstractImageMultiplier,
    ):
        self.operator1 = operator1
        self.operator2 = operator2
        self.buffer = operator1.buffer * operator2.buffer

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class SumImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the sum of two operators."""

    operator1: AbstractImageOperator | Partial
    operator2: AbstractImageOperator | Partial

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) + self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class DiffImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractImageOperator | Partial
    operator2: AbstractImageOperator | Partial

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) - self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class ProductImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractImageOperator | Partial
    operator2: AbstractImageOperator | Partial

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) * self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

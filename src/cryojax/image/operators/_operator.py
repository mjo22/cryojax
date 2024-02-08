"""
Base classes for image operators.
"""

from abc import abstractmethod
from typing import overload, Any, Callable
from typing_extensions import override
from jaxtyping import Array

import jax
import jax.numpy as jnp
from equinox import Module, field, AbstractVar

from ...typing import ImageCoords, VolumeCoords, Image, Volume, Real_


class AbstractImageOperator(Module, strict=True):
    """
    The base class for image operators that contain
    model parameters and compute an ``Array`` at runtime.
    """

    @overload
    @abstractmethod
    def __call__(
        self, coords_or_freqs: ImageCoords | VolumeCoords, **kwargs: Any
    ) -> Array: ...

    @overload
    @abstractmethod
    def __call__(self, coords_or_freqs: None, **kwargs: Any) -> Array: ...

    @abstractmethod
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError

    def __add__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return SumImageOperator(self, other)
        return SumImageOperator(self, Constant(other))

    def __radd__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return SumImageOperator(other, self)
        return SumImageOperator(Constant(other), self)

    def __sub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return DiffImageOperator(self, other)
        return DiffImageOperator(self, Constant(other))

    def __rsub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return DiffImageOperator(other, self)
        return DiffImageOperator(Constant(other), self)

    def __mul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return ProductImageOperator(self, other)
        return ProductImageOperator(self, Constant(other))

    def __rmul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
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

    buffer: AbstractVar[Image | Volume]

    def __call__(self, image: Image | Volume) -> Image | Volume:
        return image * jax.lax.stop_gradient(self.buffer)

    def __mul__(self, other) -> "AbstractImageMultiplier":
        return ProductImageMultiplier(operator1=self, operator2=other)

    def __rmul__(self, other) -> "AbstractImageMultiplier":
        return ProductImageMultiplier(operator1=other, operator2=self)


class Constant(AbstractImageOperator, strict=True):
    """An operator that is a constant."""

    value: Real_ = field(default=1.0, converter=jnp.asarray)

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Real_:
        return self.value


class Lambda(AbstractImageOperator, strict=True):
    """An operator that calls a custom function."""

    fn: Callable[[ImageCoords | VolumeCoords], Image] = field(static=True)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | VolumeCoords, **kwargs: Any
    ) -> Image:
        return self.fn(coords_or_freqs, **kwargs)


class Empirical(AbstractImageOperator, strict=True):
    r"""
    This operator stores a measured image, rather than
    computing one from a model.

    Attributes
    ----------
    measurement :
        The measured image.
    amplitude :
        An amplitude scaling for the operator.
    """

    measurement: Image | Volume

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    offset: Real_ = field(default=0.0, converter=jnp.asarray)

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Image:
        """Return the scaled and offset measurement."""
        return self.amplitude * jax.lax.stop_gradient(self.measurement)


class ProductImageMultiplier(AbstractImageMultiplier, strict=True):
    """A helper to represent the product of two operators."""

    buffer: Image | Volume

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

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) + self.operator2(coords_or_freqs)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class DiffImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) - self.operator2(coords_or_freqs)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class ProductImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) * self.operator2(coords_or_freqs)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

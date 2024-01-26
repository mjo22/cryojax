"""
Base classes for image operators.
"""

from __future__ import annotations

__all__ = [
    "AbstractImageMultiplier",
    "AbstractImageOperator",
    "ImageMultiplierT",
    "ImageOperatorT",
    "Constant",
    "Empirical",
    "Lambda",
]

from abc import abstractmethod
from typing import overload, Any, TypeVar, Callable
from typing_extensions import override
from jaxtyping import Array

import jax
from equinox import Module

from ...core import field
from ...typing import ImageCoords, VolumeCoords, Image, Volume, Real_

ImageMultiplierT = TypeVar("ImageMultiplierT", bound="AbstractImageMultiplier")
"""TypeVar for ``ProductOperatorAsBuffer``s"""

ImageOperatorT = TypeVar("ImageOperatorT", bound="AbstractImageOperator")
"""TypeVar for ``OperatorAsFunction``s"""


class AbstractImageOperator(Module):
    """
    The base class for image operators that contain
    model parameters and compute an ``Array`` at runtime.
    """

    @overload
    @abstractmethod
    def __call__(
        self, coords_or_freqs: ImageCoords | VolumeCoords, **kwargs: Any
    ) -> Array:
        ...

    @overload
    @abstractmethod
    def __call__(self, coords_or_freqs: None, **kwargs: Any) -> Array:
        ...

    @abstractmethod
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError

    def __add__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _SumImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _SumImageOperator(self, other)
        return _SumImageOperator(self, Constant(other))

    def __radd__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _SumImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _SumImageOperator(other, self)
        return _SumImageOperator(Constant(other), self)

    def __sub__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _DiffImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _DiffImageOperator(self, other)
        return _DiffImageOperator(self, Constant(other))

    def __rsub__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _DiffImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _DiffImageOperator(other, self)
        return _DiffImageOperator(Constant(other), self)

    def __mul__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _ProductImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _ProductImageOperator(self, other)
        return _ProductImageOperator(self, Constant(other))

    def __rmul__(
        self: AbstractImageOperator,
        other: AbstractImageOperator | Real_,
    ) -> _ProductImageOperator:
        if isinstance(other, AbstractImageOperator):
            return _ProductImageOperator(other, self)
        return _ProductImageOperator(Constant(other), self)


class AbstractImageMultiplier(Module):
    """
    Base class for computing and applying an ``Array`` to an image.

    Attributes
    ----------
    operator :
        The operator. Note that this is automatically
        computed upon instantiation.
    """

    buffer: Image | Volume

    def __init__(self, buffer: Image | Volume) -> None:
        """Compute the operator."""
        self.buffer = buffer

    def __call__(self, image: Image | Volume) -> Image | Volume:
        return image * jax.lax.stop_gradient(self.buffer)

    def __mul__(
        self: ImageMultiplierT, other: ImageMultiplierT
    ) -> _ProductImageMultiplier:
        return _ProductImageMultiplier(operator1=self, operator2=other)

    def __rmul__(
        self: ImageMultiplierT, other: ImageMultiplierT
    ) -> _ProductImageMultiplier:
        return _ProductImageMultiplier(operator1=other, operator2=self)


class Constant(AbstractImageOperator):
    """An operator that is a constant."""

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Real_:
        return self.value


class Lambda(AbstractImageOperator):
    """An operator that calls a custom function."""

    fn: Callable[[ImageCoords | VolumeCoords], Image] = field(static=True)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | VolumeCoords, **kwargs: Any
    ) -> Image:
        return self.fn(coords_or_freqs, **kwargs)


class Empirical(AbstractImageOperator):
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

    amplitude: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Image:
        """Return the scaled and offset measurement."""
        return self.amplitude * jax.lax.stop_gradient(self.measurement)


class _ProductImageMultiplier(AbstractImageMultiplier):
    """A helper to represent the product of two operators."""

    operator1: AbstractImageMultiplier
    operator2: AbstractImageMultiplier

    @override
    def __init__(
        self,
        operator1: ImageMultiplierT,
        operator2: ImageMultiplierT,
    ):
        self.operator1 = operator1
        self.operator2 = operator2
        self.buffer = operator1.buffer * operator2.buffer

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class _SumImageOperator(AbstractImageOperator):
    """A helper to represent the sum of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) + self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffImageOperator(AbstractImageOperator):
    """A helper to represent the difference of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) - self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductImageOperator(AbstractImageOperator):
    """A helper to represent the product of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(
        self,
        coords_or_freqs: ImageCoords | VolumeCoords | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.operator1(coords_or_freqs) * self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

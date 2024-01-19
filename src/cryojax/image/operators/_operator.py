"""
Base classes for image operators.
"""

from __future__ import annotations

__all__ = [
    "ImageMultiplier",
    "ImageOperator",
    "ImageMultiplierT",
    "ImageOperator",
    "Constant",
    "Empirical",
]

from abc import abstractmethod
from typing import overload, Any, TypeVar
from typing_extensions import override
from jaxtyping import Array

import jax
from equinox import Module

from ...core import field
from ...typing import ImageCoords, Image, Real_

ImageMultiplierT = TypeVar("ImageMultiplierT", bound="ImageMultiplier")
"""TypeVar for ``ProductOperatorAsBuffer``s"""

ImageOperatorT = TypeVar("ImageOperatorT", bound="ImageOperator")
"""TypeVar for ``OperatorAsFunction``s"""


class ImageOperator(Module):
    """
    The base class for image operators that contain
    model parameters and compute an ``Array`` at runtime.
    """

    @overload
    @abstractmethod
    def __call__(self, coords_or_freqs: ImageCoords, **kwargs: Any) -> Array:
        ...

    @overload
    @abstractmethod
    def __call__(self, coords_or_freqs: None, **kwargs: Any) -> Array:
        ...

    @abstractmethod
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError

    def __add__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _SumImageOperator:
        if isinstance(other, ImageOperator):
            return _SumImageOperator(self, other)
        return _SumImageOperator(self, Constant(other))

    def __radd__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _SumImageOperator:
        if isinstance(other, ImageOperator):
            return _SumImageOperator(other, self)
        return _SumImageOperator(Constant(other), self)

    def __sub__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _DiffImageOperator:
        if isinstance(other, ImageOperator):
            return _DiffImageOperator(self, other)
        return _DiffImageOperator(self, Constant(other))

    def __rsub__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _DiffImageOperator:
        if isinstance(other, ImageOperator):
            return _DiffImageOperator(other, self)
        return _DiffImageOperator(Constant(other), self)

    def __mul__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _ProductImageOperator:
        if isinstance(other, ImageOperator):
            return _ProductImageOperator(self, other)
        return _ProductImageOperator(self, Constant(other))

    def __rmul__(
        self: ImageOperator,
        other: ImageOperator | Real_,
    ) -> _ProductImageOperator:
        if isinstance(other, ImageOperator):
            return _ProductImageOperator(other, self)
        return _ProductImageOperator(Constant(other), self)


class ImageMultiplier(Module):
    """
    Base class for computing and applying an ``Array`` to an image.

    Attributes
    ----------
    operator :
        The operator. Note that this is automatically
        computed upon instantiation.
    """

    buffer: Image

    def __init__(self, buffer: Image) -> None:
        """Compute the operator."""
        self.buffer = buffer

    def __call__(self, image: Image) -> Image:
        return image * jax.lax.stop_gradient(self.buffer)

    def __mul__(
        self: ImageMultiplierT, other: ImageMultiplierT
    ) -> _ProductImageMultiplier:
        return _ProductImageMultiplier(operator1=self, operator2=other)

    def __rmul__(
        self: ImageMultiplierT, other: ImageMultiplierT
    ) -> _ProductImageMultiplier:
        return _ProductImageMultiplier(operator1=other, operator2=self)


class Constant(ImageOperator):
    """An operator that is a constant."""

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        return self.value


class Empirical(ImageOperator):
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

    measurement: Image

    amplitude: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Image:
        """Return the scaled and offset measurement."""
        return self.amplitude * jax.lax.stop_gradient(self.measurement)


class _ProductImageMultiplier(ImageMultiplier):
    """A helper to represent the product of two operators."""

    operator1: ImageMultiplier
    operator2: ImageMultiplier

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


class _SumImageOperator(ImageOperator):
    """A helper to represent the sum of two operators."""

    operator1: ImageOperator
    operator2: ImageOperator

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords_or_freqs) + self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffImageOperator(ImageOperator):
    """A helper to represent the difference of two operators."""

    operator1: ImageOperator
    operator2: ImageOperator

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords_or_freqs) - self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductImageOperator(ImageOperator):
    """A helper to represent the product of two operators."""

    operator1: ImageOperator
    operator2: ImageOperator

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords_or_freqs) * self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

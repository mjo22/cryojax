"""
Base classes for image operators.
"""

from __future__ import annotations

__all__ = [
    "OperatorAsBuffer",
    "OperatorAsFunction",
    "OperatorAsBufferT",
    "OperatorAsFunctionT",
    "OperatorLike",
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

OperatorAsBufferT = TypeVar("OperatorAsBufferT", bound="OperatorAsBuffer")
"""TypeVar for ``OperatorAsBuffer``s"""

OperatorAsFunctionT = TypeVar(
    "OperatorAsFunctionT", bound="OperatorAsFunction"
)
"""TypeVar for ``OperatorAsFunction``s"""


class OperatorAsBuffer(Module):
    """
    Base class for computing and applying an ``Array`` to an image.

    Attributes
    ----------
    operator :
        The operator. Note that this is automatically
        computed upon instantiation.
    """

    operator: Image

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Compute the operator."""
        super().__init__(**kwargs)

    def get(self, *args: Any, **kwargs: Any):
        """Get the operator."""
        return self.operator

    def __call__(self, image: Image) -> Image:
        """Apply the operator to an image."""
        return self.operator * image

    def __mul__(
        self: OperatorAsBufferT, other: OperatorAsBufferT
    ) -> _ProductOperatorAsBuffer:
        return _ProductOperatorAsBuffer(operator1=self, operator2=other)

    def __rmul__(
        self: OperatorAsBufferT, other: OperatorAsBufferT
    ) -> _ProductOperatorAsBuffer:
        return _ProductOperatorAsBuffer(operator1=other, operator2=self)

    def __add__(
        self: OperatorAsBufferT, other: OperatorAsBufferT
    ) -> _SumOperatorAsBuffer:
        return _SumOperatorAsBuffer(operator1=self, operator2=other)

    def __radd__(
        self: OperatorAsBufferT, other: OperatorAsBufferT
    ) -> _SumOperatorAsBuffer:
        return _SumOperatorAsBuffer(operator1=other, operator2=self)


class OperatorAsFunction(Module):
    """
    The base class for image operators that contain
    model parameters and compute an ``Array`` at runtime.
    """

    def get(self, *args: Any, **kwargs: Any):
        """Get the operator."""
        return self.__call__(*args, **kwargs)

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
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _SumOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _SumOperatorAsFunction(self, other)
        return _SumOperatorAsFunction(self, Constant(other))

    def __radd__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _SumOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _SumOperatorAsFunction(other, self)
        return _SumOperatorAsFunction(Constant(other), self)

    def __mul__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _ProductOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _ProductOperatorAsFunction(self, other)
        return _ProductOperatorAsFunction(self, Constant(other))

    def __rmul__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _ProductOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _ProductOperatorAsFunction(other, self)
        return _ProductOperatorAsFunction(Constant(other), self)


OperatorLike = OperatorAsBuffer | OperatorAsFunction


class Constant(OperatorAsFunction):
    """An operator that is a constant."""

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        return self.value


class Empirical(OperatorAsFunction):
    r"""
    This operator stores a measured image, rather than
    computing one from a model.

    Attributes
    ----------
    measurement :
        The measured image.
    amplitude :
        An amplitude scaling for the operator.
    offset :
        An offset added to the above equation.
    """

    measurement: Image

    amplitude: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Image:
        """Return the scaled and offset measurement."""
        return (
            self.amplitude * jax.lax.stop_gradient(self.measurement)
            + self.offset
        )


class _SumOperatorAsBuffer(OperatorAsBuffer):
    """A helper to represent the product of two operators."""

    operator1: OperatorAsBuffer
    operator2: OperatorAsBuffer

    @override
    def __init__(
        self,
        operator1: OperatorAsBufferT,
        operator2: OperatorAsBufferT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.operator1 = operator1
        self.operator2 = operator2
        self.operator = operator1.operator + operator2.operator

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _ProductOperatorAsBuffer(OperatorAsBuffer):
    """A helper to represent the product of two operators."""

    operator1: OperatorAsBuffer
    operator2: OperatorAsBuffer

    @override
    def __init__(
        self,
        operator1: OperatorAsBufferT,
        operator2: OperatorAsBufferT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.operator1 = operator1
        self.operator2 = operator2
        self.operator = operator1.operator * operator2.operator

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class _SumOperatorAsFunction(OperatorAsFunction):
    """A helper to represent the sum of two operators."""

    operator1: OperatorAsFunction
    operator2: OperatorAsFunction

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords_or_freqs) + self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _ProductOperatorAsFunction(OperatorAsFunction):
    """A helper to represent the product of two operators."""

    operator1: OperatorAsFunction
    operator2: OperatorAsFunction

    @override
    def __call__(
        self, coords_or_freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords_or_freqs) * self.operator2(
            coords_or_freqs
        )

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

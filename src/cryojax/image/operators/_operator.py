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
]

from abc import abstractmethod
from typing import overload, Any, TypeVar
from typing_extensions import override
from jaxtyping import Array

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
    def __call__(self, coords: ImageCoords, **kwargs: Any) -> Array:
        ...

    @overload
    @abstractmethod
    def __call__(self, coords: None, **kwargs: Any) -> Array:
        ...

    @abstractmethod
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError

    def __add__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _SumOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _SumOperatorAsFunction(self, other)
        return _SumOperatorAsFunction(self, _Constant(other))

    def __radd__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _SumOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _SumOperatorAsFunction(other, self)
        return _SumOperatorAsFunction(_Constant(other), self)

    def __mul__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _ProductOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _ProductOperatorAsFunction(self, other)
        return _ProductOperatorAsFunction(self, _Constant(other))

    def __rmul__(
        self: OperatorAsFunctionT,
        other: OperatorAsFunctionT | Real_,
    ) -> _ProductOperatorAsFunction:
        if isinstance(other, OperatorAsFunction):
            return _ProductOperatorAsFunction(other, self)
        return _ProductOperatorAsFunction(_Constant(other), self)


OperatorLike = OperatorAsBuffer | OperatorAsFunction


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


class _Constant(OperatorAsFunction):
    """A helper to add a constant to an operator."""

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        return self.value


class _SumOperatorAsFunction(OperatorAsFunction):
    """A helper to represent the sum of two operators."""

    operator1: OperatorAsFunction
    operator2: OperatorAsFunction

    @override
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords) + self.operator2(coords)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _ProductOperatorAsFunction(OperatorAsFunction):
    """A helper to represent the product of two operators."""

    operator1: OperatorAsFunction
    operator2: OperatorAsFunction

    @override
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.operator1(coords) * self.operator2(coords)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"

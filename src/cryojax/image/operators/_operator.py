"""
Base classes for image operators.
"""

from __future__ import annotations

__all__ = [
    "ImageOperator",
    "ParameterizedImageOperator",
    "Constant",
    "ImageOperatorT",
    "ParameterizedImageOperatorT",
    "OperatorLike",
]

from abc import abstractmethod
from typing import overload, Any, TypeVar
from typing_extensions import override
from jaxtyping import Array

import jax.numpy as jnp
from equinox import Module

from ...core import field
from ...typing import ImageCoords, Image, Real_

ImageOperatorT = TypeVar("ImageOperatorT", bound="ImageOperator")
"""TypeVar for ``ImageOperator``s"""

ParameterizedImageOperatorT = TypeVar(
    "ParameterizedImageOperatorT", bound="ParameterizedImageOperator"
)
"""TypeVar for ``ParameterizedImageOperator``s"""


class ImageOperator(Module):
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

    def __call__(self, image: Image) -> Image:
        """Apply the operator to an image."""
        return self.operator * image

    def __mul__(
        self: ImageOperatorT, other: ImageOperatorT
    ) -> _ProductOperator:
        return _ProductOperator(operator1=self, operator2=other)

    def __rmul__(
        self: ImageOperatorT, other: ImageOperatorT
    ) -> _ProductOperator:
        return _ProductOperator(operator1=other, operator2=self)


class _ProductOperator(ImageOperator):
    """A helper to represent the product of two operators."""

    operator1: ImageOperator
    operator2: ImageOperator

    @override
    def __init__(
        self,
        operator1: ImageOperatorT,
        operator2: ImageOperatorT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.operator1 = operator1
        self.operator2 = operator2
        self.operator = operator1.operator * operator2.operator

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class ParameterizedImageOperator(Module):
    """
    The base class for image operators that contain
    model parameters and are computed at runtime.

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite :func:`Kernel.__call__`.
    """

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
        self: ParameterizedImageOperatorT,
        other: ParameterizedImageOperatorT | Real_,
    ) -> _SumParameterizedOperator:
        if isinstance(other, ParameterizedImageOperator):
            return _SumParameterizedOperator(self, other)
        return _SumParameterizedOperator(self, Constant(other))

    def __radd__(
        self: ParameterizedImageOperatorT,
        other: ParameterizedImageOperatorT | Real_,
    ) -> _SumParameterizedOperator:
        if isinstance(other, ParameterizedImageOperator):
            return _SumParameterizedOperator(other, self)
        return _SumParameterizedOperator(Constant(other), self)

    def __mul__(
        self: ParameterizedImageOperatorT,
        other: ParameterizedImageOperatorT | Real_,
    ) -> _ProductParameterizedOperator:
        if isinstance(other, ParameterizedImageOperator):
            return _ProductParameterizedOperator(self, other)
        return _ProductParameterizedOperator(self, Constant(other))

    def __rmul__(
        self: ParameterizedImageOperatorT,
        other: ParameterizedImageOperatorT | Real_,
    ) -> _ProductParameterizedOperator:
        if isinstance(other, ParameterizedImageOperator):
            return _ProductParameterizedOperator(other, self)
        return _ProductParameterizedOperator(Constant(other), self)


class Constant(ParameterizedImageOperator):
    """
    This kernel returns a constant.

    Attributes
    ----------
    value :
        The value of the kernel.
    """

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.value


class _SumParameterizedOperator(ParameterizedImageOperator):
    """A helper to represent the sum of two kernels"""

    kernel1: ParameterizedImageOperator
    kernel2: ParameterizedImageOperator

    @override
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.kernel1(coords) + self.kernel2(coords)

    def __repr__(self):
        return f"{repr(self.kernel1)} + {repr(self.kernel2)}"


class _ProductParameterizedOperator(ParameterizedImageOperator):
    """A helper to represent the product of two kernels"""

    kernel1: ParameterizedImageOperator
    kernel2: ParameterizedImageOperator

    @override
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.kernel1(coords) * self.kernel2(coords)

    def __repr__(self):
        return f"{repr(self.kernel1)} * {repr(self.kernel2)}"


OperatorLike = ImageOperator | ParameterizedImageOperator

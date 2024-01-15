"""
Implementation of a RealOperator.
"""

from __future__ import annotations

__all__ = ["RealOperator", "Gaussian", "RealOperatorLike"]

from abc import abstractmethod
from typing import Any
from typing_extensions import override
from jaxtyping import Array

import jax.numpy as jnp

from ._operator import OperatorAsFunction
from ...core import field
from ...typing import ImageCoords, RealImage, Real_


class RealOperator(OperatorAsFunction):
    """
    The base class for all real operators.

    By convention, operators should be defined to
    have units of inverse area (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @abstractmethod
    def __call__(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError


RealOperatorLike = RealOperator | OperatorAsFunction


class Gaussian(RealOperator):
    r"""
    This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        g(r) = \frac{\kappa}{2\pi \beta} \exp(- r^2 / (2 \beta)),

    where :math:`r^2 = x^2 + y^2`.

    Attributes
    ----------
    amplitude :
        The amplitude of the operator, equal to :math:`\kappa`
        in the above equation.
    b_factor :
        The length scale of the operator, equal to :math:`\beta`
        in the above equation.
    offset :
        An offset added to the above equation.
    """

    amplitude: Real_ = field(default=1.0)
    b_factor: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(self, coords: ImageCoords | None, **kwargs: Any) -> RealImage:
        if coords is None:
            raise ValueError(
                "The coordinate grid must be given as an argument to the operator call."
            )
        else:
            r_sqr = jnp.sum(coords**2, axis=-1)
            scaling = (
                self.amplitude / (2 * jnp.pi * self.b_factor)
            ) * jnp.exp(-0.5 * r_sqr / self.b_factor)
            return scaling + self.offset

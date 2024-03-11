"""
Implementation of operators on images in real-space.
"""

from abc import abstractmethod
from typing import overload
from typing_extensions import override
from jaxtyping import Array, Float
from equinox import field

import jax.numpy as jnp

from ._operator import AbstractImageOperator
from ...typing import ImageCoords, VolumeCoords, RealImage, RealVolume, Real_
from ...core import error_if_negative


class AbstractRealOperator(AbstractImageOperator, strict=True):
    """
    The base class for all real operators.

    By convention, operators should be defined to
    have units of inverse area (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @overload
    @abstractmethod
    def __call__(self, coords: ImageCoords) -> RealImage: ...

    @overload
    @abstractmethod
    def __call__(self, coords: VolumeCoords) -> RealVolume: ...

    @abstractmethod
    def __call__(self, coords: ImageCoords | VolumeCoords) -> RealImage | RealVolume:
        raise NotImplementedError


RealOperatorLike = AbstractRealOperator | AbstractImageOperator


class Gaussian2D(AbstractRealOperator, strict=True):
    r"""
    This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        g(r) = \frac{\kappa}{2\pi \beta} \exp(- (r - r_0)^2 / (2 \beta)),

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
        An offset to the origin, equal to :math:`r_0`
        in the above equation.
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    b_factor: Real_ = field(default=1.0, converter=error_if_negative)
    offset: Float[Array, "2"] = field(default=(0.0, 0.0), converter=jnp.asarray)

    @override
    def __call__(self, coords: ImageCoords) -> RealImage:
        r_sqr = jnp.sum((coords - self.offset) ** 2, axis=-1)
        scaling = (self.amplitude / jnp.sqrt(2 * jnp.pi * self.b_factor)) * jnp.exp(
            -0.5 * r_sqr / self.b_factor
        )
        return scaling

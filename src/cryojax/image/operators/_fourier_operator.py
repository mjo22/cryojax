"""
Implementation of a FourierOperator. Put simply, these are
functions commonly applied to images in fourier space.

Opposed to a Filter, a FourierOperator is computed at
runtime---not upon initialization.

These classes are modified from the library ``tinygp``.
"""

from __future__ import annotations

__all__ = [
    "FourierOperator",
    "Constant",
    "ZeroMode",
    "Exp",
    "Gaussian",
    "Empirical",
]

from abc import abstractmethod
from typing import Any
from typing_extensions import override
from jaxtyping import Array

import jax.numpy as jnp

from ._operator import OperatorAsFunction
from ...core import field
from ...typing import Real_, ImageCoords, RealImage, Image


class FourierOperator(OperatorAsFunction):
    """
    The base class for all fourier-based operators.

    By convention, operators should be defined to
    be dimensionless (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @abstractmethod
    def __call__(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError


class Constant(OperatorAsFunction):
    """An operator that is a constant."""

    value: Real_ = field(default=1.0)

    @override
    def __call__(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        return self.value


class ZeroMode(FourierOperator):
    """
    This operator returns a constant in the zero mode.

    Attributes
    ----------
    value :
        The value of the zero mode.
    """

    value: Real_ = field(default=1.0)

    @override
    def __call__(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the operator call."
            )
        else:
            N1, N2 = freqs.shape[0:-1]
            return jnp.zeros((N1, N2)).at[0, 0].set(N1 * N2 * self.value)


class Exp(FourierOperator):
    r"""
    This operator, in real space, represents a covariance
    function equal to an exponential decay, given by

    .. math::
        g(r) = \frac{\kappa}{2 \pi \xi^2} \exp(- r / \xi),

    where :math:`r = \sqrt{x^2 + y^2}` is a radial coordinate.
    Here, :math:`\xi` has dimensions of length and :math:`g(r)`
    has dimensions of inverse area. The power spectrum from such
    a correlation function (in two-dimensions) is given by its
    Hankel transform pair

    .. math::
        P(k) = \frac{\kappa}{2 \pi \xi^3} \frac{1}{(\xi^{-2} + k^2)^{3/2}}.

    Here :math:`\kappa` is a scale factor.

    Attributes
    ----------
    amplitude :
        The amplitude of the operator, equal to :math:`\kappa`
        in the above equation. Note that this has dimensions
        of inverse volume.
    scale :
        The length scale of the operator, equal to :math:`\xi`
        in the above equation.
    offset :
        An offset added to the above equation.
    """

    amplitude: Real_ = field(default=1.0)
    scale: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the operator call."
            )
        else:
            k_sqr = jnp.sum(freqs**2, axis=-1)
            scaling = 1.0 / (k_sqr + jnp.divide(1, (self.scale) ** 2)) ** 1.5
            scaling *= jnp.divide(self.amplitude, 2 * jnp.pi * self.scale**3)
            return scaling + self.offset


class Gaussian(FourierOperator):
    r"""
    This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 2),

    where :math:`k^2 = k_x^2 + k_y^2` is the length of the
    wave vector. Here, :math:`\beta` has dimensions of length
    squared. The real-space version of this function is given
    by

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
    def __call__(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the operator call."
            )
        else:
            k_sqr = jnp.sum(freqs**2, axis=-1)
            scaling = self.amplitude * jnp.exp(-0.5 * self.b_factor * k_sqr)
            return scaling + self.offset


class Empirical(FourierOperator):
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

    measurement: Image = field(static=True)

    amplitude: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    @override
    def __call__(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Image:
        """Return the scaled and offset measurement."""
        return self.amplitude * self.measurement + self.offset

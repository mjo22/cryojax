"""
Implementation of an AbstractFourierOperator. Put simply, these are
functions commonly applied to images in fourier space.

Opposed to a AbstractFilter, a AbstractFourierOperator is computed at
runtime---not upon initialization. AbstractFourierOperators also do not
have a rule for how they should be applied to images.

These classes are modified from the library ``tinygp``.
"""

from abc import abstractmethod
from typing import overload
from typing_extensions import override
from equinox import field

import jax.numpy as jnp

from ._operator import AbstractImageOperator
from ...typing import (
    Real_,
    ImageCoords,
    VolumeCoords,
    RealImage,
    RealVolume,
    Image,
    Volume,
)


class AbstractFourierOperator(AbstractImageOperator, strict=True):
    """
    The base class for all fourier-based operators.

    By convention, operators should be defined to
    be dimensionless (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @overload
    @abstractmethod
    def __call__(self, freqs: ImageCoords) -> Image: ...

    @overload
    @abstractmethod
    def __call__(self, freqs: VolumeCoords) -> Volume: ...

    @abstractmethod
    def __call__(self, freqs: ImageCoords | VolumeCoords) -> Image | Volume:
        raise NotImplementedError


FourierOperatorLike = AbstractFourierOperator | AbstractImageOperator


class ZeroMode(AbstractFourierOperator, strict=True):
    """
    This operator returns a constant in the zero mode.

    Attributes
    ----------
    value :
        The value of the zero mode.
    """

    value: Real_ = field(default=0.0, converter=jnp.asarray)

    @override
    def __call__(self, freqs: ImageCoords) -> RealImage:
        N1, N2 = freqs.shape[0:-1]
        return jnp.zeros((N1, N2)).at[0, 0].set(self.value)


class FourierExp2D(AbstractFourierOperator, strict=True):
    r"""
    This operator, in real space, represents a
    function equal to an exponential decay, given by

    .. math::
        g(|r|) = \frac{\kappa}{2 \pi \xi^2} \exp(- |r| / \xi),

    where :math:`|r| = \sqrt{x^2 + y^2}` is a radial coordinate.
    Here, :math:`\xi` has dimensions of length and :math:`g(r)`
    has dimensions of inverse area. The power spectrum from such
    a correlation function (in two-dimensions) is given by its
    Hankel transform pair

    .. math::
        P(|k|) = \frac{\kappa}{2 \pi \xi^3} \frac{1}{(\xi^{-2} + |k|^2)^{3/2}}.

    Here :math:`\kappa` is a scale factor and :math:`\xi` is a length
    scale.

    Attributes
    ----------
    amplitude :
        The amplitude of the operator, equal to :math:`\kappa`
        in the above equation.
    scale :
        The length scale of the operator, equal to :math:`\xi`
        in the above equation.
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    scale: Real_ = field(default=1.0, converter=jnp.asarray)

    @override
    def __call__(self, freqs: ImageCoords) -> RealImage:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, (self.scale) ** 2)) ** 1.5
        scaling *= jnp.divide(self.amplitude, 2 * jnp.pi * self.scale**3)
        return scaling


class Lorenzian(AbstractFourierOperator, strict=True):
    r"""
    This operator is the Lorenzian, given

    .. math::
        P(|k|) = \frac{\kappa}{\xi^2} \frac{1}{(\xi^{-2} + |k|^2)}.

    Here :math:`\kappa` is a scale factor and :math:`\xi` is a length
    scale.

    Attributes
    ----------
    amplitude :
        The amplitude of the operator, equal to :math:`\kappa`
        in the above equation.
    scale :
        The length scale of the operator, equal to :math:`\xi`
        in the above equation.
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    scale: Real_ = field(default=1.0, converter=jnp.asarray)

    @overload
    def __call__(self, freqs: ImageCoords) -> RealImage: ...

    @overload
    def __call__(self, freqs: VolumeCoords) -> RealVolume: ...

    @override
    def __call__(self, freqs: ImageCoords | VolumeCoords) -> RealImage | RealVolume:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, self.scale**2))
        scaling *= jnp.divide(self.amplitude, self.scale**2)
        return scaling


class FourierGaussian(AbstractFourierOperator, strict=True):
    r"""
    This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 2),

    where :math:`k^2 = k_x^2 + k_y^2` is the length of the
    wave vector. Here, :math:`\beta` has dimensions of length
    squared. In 2D, the real-space version of this function is given
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
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    b_factor: Real_ = field(default=1.0, converter=jnp.asarray)

    @overload
    def __call__(self, freqs: ImageCoords) -> RealImage: ...

    @overload
    def __call__(self, freqs: VolumeCoords) -> RealVolume: ...

    @override
    def __call__(self, freqs: ImageCoords | VolumeCoords) -> RealImage | RealVolume:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = self.amplitude * jnp.exp(-0.5 * self.b_factor * k_sqr)
        return scaling

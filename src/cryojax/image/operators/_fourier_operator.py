"""
Implementation of an AbstractFourierOperator. Put simply, these are
functions commonly applied to images in fourier space.

Opposed to a AbstractFilter, a AbstractFourierOperator is computed at
runtime---not upon initialization. AbstractFourierOperators also do not
have a rule for how they should be applied to images.

These classes are modified from the library ``tinygp``.
"""

from abc import abstractmethod
from typing import Any, Optional
from typing_extensions import override
from jaxtyping import Array
from equinox import field

import jax.numpy as jnp

from ._operator import AbstractImageOperator
from ...typing import Real_, ImageCoords, VolumeCoords, RealImage


class AbstractFourierOperator(AbstractImageOperator):
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
        self, freqs: ImageCoords | VolumeCoords | None = None, **kwargs: Any
    ) -> Array:
        raise NotImplementedError


FourierOperatorLike = AbstractFourierOperator | AbstractImageOperator


class ZeroMode(AbstractFourierOperator):
    """
    This operator returns a constant in the zero mode.

    Attributes
    ----------
    value :
        The value of the zero mode.
    """

    value: Real_ = field(default=0.0, converter=jnp.asarray)

    @override
    def __call__(
        self,
        freqs: ImageCoords | None,
        half_space: bool = True,
        shape_in_real_space: Optional[None] = None,
        **kwargs: Any,
    ) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the operator call."
            )
        else:
            N1, N2 = freqs.shape[0:-1]
            if half_space:
                N_modes = (
                    N1 * (2 * N2 - 1)
                    if shape_in_real_space is None
                    else shape_in_real_space[0] * shape_in_real_space[1]
                )
            else:
                N_modes = N1 * N2
            return jnp.zeros((N1, N2)).at[0, 0].set(N_modes * self.value)


class FourierExp2D(AbstractFourierOperator):
    r"""
    This operator, in real space, represents a
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
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    scale: Real_ = field(default=1.0, converter=jnp.asarray)

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
            return scaling


class FourierGaussian2D(AbstractFourierOperator):
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
    """

    amplitude: Real_ = field(default=1.0, converter=jnp.asarray)
    b_factor: Real_ = field(default=1.0, converter=jnp.asarray)

    @override
    def __call__(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the operator call."
            )
        else:
            k_sqr = jnp.sum(freqs**2, axis=-1)
            scaling = self.amplitude * jnp.exp(-0.5 * self.b_factor * k_sqr)
            return scaling

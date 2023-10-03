"""
Implementation of a Kernel. Put simply, these are
functions commonly applied to images, usually in fourier space.

These classes are modified from the library ``tinygp``.
"""

from __future__ import annotations

__all__ = [
    "Kernel",
    "Product",
    "Sum",
    "Constant",
    "Exp",
    "Gaussian",
    "Empirical",
    "Custom",
]

from abc import abstractmethod
from typing import Any, Union, Callable, Concatenate, ParamSpec, Optional
from jaxtyping import Array

import jax.numpy as jnp

from ..core import field, Module
from ..types import Real_, ImageCoords, RealImage, Image

P = ParamSpec("P")


class Kernel(Module):
    """
    The base class for all kernels.

    By convention, Kernels should be defined to
    be dimensionless (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite :func:`Kernel.evaluate`.
    """

    @abstractmethod
    def evaluate(self, coords: ImageCoords, **kwargs: Any) -> Array:
        """
        Evaluate the kernel at a set of coordinates.

        Arguments
        ----------
        coords :
            The real or fourier space cartesian coordinates.
        """
        pass

    def __call__(
        self, coords: ImageCoords, *args: Any, **kwargs: Any
    ) -> Array:
        coords = jnp.asarray(coords)
        return self.evaluate(coords, *args, **kwargs)

    def __add__(self, other: Union[Kernel, float]) -> Kernel:
        if isinstance(other, Kernel):
            return Sum(self, other)
        return Sum(self, Constant(other))

    def __radd__(self, other: Any) -> Kernel:
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if isinstance(other, Kernel):
            return Sum(other, self)
        return Sum(Constant(other), self)

    def __mul__(self, other: Union[Kernel, float]) -> Kernel:
        if isinstance(other, Kernel):
            return Product(self, other)
        return Product(self, Constant(other))

    def __rmul__(self, other: Any) -> Kernel:
        if isinstance(other, Kernel):
            return Product(other, self)
        return Product(Constant(other), self)


class Sum(Kernel):
    """A helper to represent the sum of two kernels"""

    kernel1: Kernel = field()
    kernel2: Kernel = field()

    def evaluate(self, coords: ImageCoords) -> Array:
        return self.kernel1(coords) + self.kernel2(coords)


class Product(Kernel):
    """A helper to represent the product of two kernels"""

    kernel1: Kernel = field()
    kernel2: Kernel = field()

    def evaluate(self, coords: ImageCoords) -> Array:
        return self.kernel1(coords) * self.kernel2(coords)


class Constant(Kernel):
    """
    This kernel returns a constant.

    Attributes
    ----------
    value :
        The value of the kernel.
    """

    value: Real_ = field(default=1.0)

    def evaluate(self, coords: Optional[ImageCoords] = None) -> Real_:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.value


class Exp(Kernel):
    r"""
    This kernel, in real space, represents a covariance
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
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation. Note that this has dimensions
        of inverse volume.
    scale :
        The length scale of the kernel, equal to :math:`\xi`
        in the above equation.
    offset :
        An offset added to the above equation.
    """

    amplitude: Real_ = field(default=1.0)
    scale: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    def evaluate(self, freqs: ImageCoords) -> RealImage:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, (self.scale) ** 2)) ** 1.5
        scaling *= jnp.divide(self.amplitude, 2 * jnp.pi * self.scale**3)
        return scaling + self.offset


class Gaussian(Kernel):
    r"""
    This kernel represents a simple gaussian.
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
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation.
    b_factor :
        The length scale of the kernel, equal to :math:`\beta`
        in the above equation.
    offset :
        An offset added to the above equation.
    """

    amplitude: Real_ = field(default=1.0)
    b_factor: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    def evaluate(self, freqs: ImageCoords) -> RealImage:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = self.amplitude * jnp.exp(-0.5 * self.b_factor * k_sqr)
        return scaling + self.offset


class Empirical(Kernel):
    r"""
    This kernel stores a measured image, rather than
    computing one from a model.

    Attributes
    ----------
    measurement :
        The measured image.
    amplitude :
        An amplitude scaling for the kernel.
    offset :
        An offset added to the above equation.
    """

    measurement: Image = field(static=True)

    amplitude: Real_ = field(default=1.0)
    offset: Real_ = field(default=0.0)

    def evaluate(self, coords: Optional[ImageCoords] = None) -> Image:
        """Return the scaled and offset measurement."""
        return self.amplitude * self.measurement + self.offset


class Custom(Kernel):
    """
    A custom kernel class implemented as a callable.

    Attributes
    ----------
    function: `Callable`
        A callable with a signature and behavior that matches
        :func:`Kernel.evaluate`.
    """

    function: Callable[Concatenate[ImageCoords, P], Array] = field(static=True)

    def evaluate(
        self, coords: ImageCoords, *args: P.args, **kwargs: P.kwargs
    ) -> Array:
        return self.function(coords, *args, **kwargs)

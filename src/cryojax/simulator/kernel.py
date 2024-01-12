"""
Implementation of a Kernel. Put simply, these are
functions commonly applied to images, usually in fourier space.

These classes are modified from the library ``tinygp``.
"""

from __future__ import annotations

__all__ = [
    "Kernel",
    "KernelType",
    "Constant",
    "Exp",
    "Gaussian",
    "Empirical",
    "Custom",
]

from abc import abstractmethod
from typing import overload, Any, Union, Callable, Concatenate, TypeVar
from typing_extensions import override
from jaxtyping import Array
from equinox import Module

import jax.numpy as jnp

from ..core import field
from ..typing import Real_, ImageCoords, RealImage, Image

KernelType = TypeVar("KernelType", bound="Kernel")
"""TypeVar for the Kernel base class"""


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

    @overload
    @abstractmethod
    def evaluate(self, freqs: ImageCoords, **kwargs: Any) -> Array:
        ...

    @overload
    @abstractmethod
    def evaluate(self, freqs: None, **kwargs: Any) -> Array:
        ...

    @abstractmethod
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> Array:
        """
        Evaluate the kernel at a set of coordinates.

        Arguments
        ----------
        coords :
            The real or fourier space cartesian coordinates.
        """
        pass

    def __call__(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.evaluate(freqs, **kwargs)

    def __add__(self, other: Union[KernelType, Real_]) -> _SumKernel:
        if isinstance(other, Kernel):
            return _SumKernel(self, other)
        return _SumKernel(self, Constant(other))

    def __radd__(self, other: Any) -> _SumKernel:
        if isinstance(other, Kernel):
            return _SumKernel(other, self)
        return _SumKernel(Constant(other), self)

    def __mul__(self, other: Union[Kernel, Real_]) -> _ProductKernel:
        if isinstance(other, Kernel):
            return _ProductKernel(self, other)
        return _ProductKernel(self, Constant(other))

    def __rmul__(self, other: Any) -> _ProductKernel:
        if isinstance(other, Kernel):
            return _ProductKernel(other, self)
        return _ProductKernel(Constant(other), self)


class _SumKernel(Kernel):
    """A helper to represent the sum of two kernels"""

    kernel1: KernelType = field()  # type: ignore
    kernel2: KernelType = field()  # type: ignore

    @override
    def evaluate(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.kernel1(freqs) + self.kernel2(freqs)

    def __repr__(self):
        return f"{repr(self.kernel1)} + {repr(self.kernel2)}"


class _ProductKernel(Kernel):
    """A helper to represent the product of two kernels"""

    kernel1: KernelType = field()  # type: ignore
    kernel2: KernelType = field()  # type: ignore

    @override
    def evaluate(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Array:
        return self.kernel1(freqs) * self.kernel2(freqs)

    def __repr__(self):
        return f"{repr(self.kernel1)} * {repr(self.kernel2)}"


class Constant(Kernel):
    """
    This kernel returns a constant.

    Attributes
    ----------
    value :
        The value of the kernel.
    """

    value: Real_ = field(default=1.0)

    @override
    def evaluate(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Real_:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.value


class ZeroMode(Kernel):
    """
    This kernel returns a constant in the zero mode.

    Attributes
    ----------
    value :
        The value of the zero mode.
    """

    value: Real_ = field(default=1.0)

    @override
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the Kernel call."
            )
        else:
            N1, N2 = freqs.shape[0:-1]
            return jnp.zeros((N1, N2)).at[0, 0].set(N1 * N2 * self.value)


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

    @override
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the Kernel call."
            )
        else:
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

    @override
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the Kernel call."
            )
        else:
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

    @override
    def evaluate(
        self, freqs: ImageCoords | None = None, **kwargs: Any
    ) -> Image:
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

    function: Callable[Concatenate[ImageCoords, ...], Array] = field(
        static=True
    )

    @override
    def evaluate(
        self, freqs: ImageCoords | None = None, *args: Any, **kwargs: Any
    ) -> Array:
        if freqs is None:
            return self.function(*args, **kwargs)
        else:
            return self.function(freqs, *args, **kwargs)

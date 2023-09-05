"""
Implementation of a Kernel. Put simply, these are fourier-space 
functions commonly applied to images. The word "kernel" is
borrowed from the theory of gaussian processes, where a kernel
is jargon for a covariance function. One can think of this
implementation as a fourier-space version of any stationary kernel,
since these yield a diagonal covariance in fourier space.

These classes are modified from the library tinygp.
"""

from __future__ import annotations

__all__ = [
    "Kernel",
    "Product",
    "Sum",
    "Constant",
    "Exp",
    "ExpSquared",
    "Empirical",
    "Custom",
]

from abc import ABCMeta, abstractmethod
from typing import Any, Union, Callable, Concatenate, ParamSpec
from functools import partial

import jax.numpy as jnp

from ..core import dataclass, field, Float, ArrayLike, Array, CryojaxObject

P = ParamSpec("P")


@partial(dataclass, kw_only=True)
class Kernel(CryojaxObject, metaclass=ABCMeta):
    """
    The base class for all kernels.

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite :func:`Kernel.evaluate`.
    """

    @abstractmethod
    def evaluate(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        """
        Evaluate the kernel at a set of frequencies.
        """
        pass

    def __call__(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        freqs = jnp.asarray(freqs)
        return self.evaluate(freqs, **kwargs)

    def __add__(self, other: Union[Kernel, Float]) -> Kernel:
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

    def __mul__(self, other: Union[Kernel, Float]) -> Kernel:
        if isinstance(other, Kernel):
            return Product(self, other)
        return Product(self, Constant(other))

    def __rmul__(self, other: Any) -> Kernel:
        if isinstance(other, Kernel):
            return Product(other, self)
        return Product(Constant(other), self)


@dataclass
class Sum(Kernel):
    """A helper to represent the sum of two kernels"""

    kernel1: Kernel = field(pytree_node=False)
    kernel2: Kernel = field(pytree_node=False)

    def evaluate(self, freqs: ArrayLike) -> Array:
        return self.kernel1.evaluate(freqs) + self.kernel2.evaluate(freqs)


@dataclass
class Product(Kernel):
    """A helper to represent the product of two kernels"""

    kernel1: Kernel = field(pytree_node=False)
    kernel2: Kernel = field(pytree_node=False)

    def evaluate(self, freqs: ArrayLike) -> Array:
        return self.kernel1.evaluate(freqs) * self.kernel2.evaluate(freqs)


@dataclass
class Constant(Kernel):
    """
    This kernel returns a constant.

    Attributes
    ----------
    constant : `cryojax.core.Float`
        The value of the kernel.
    """

    constant: Float = field(pytree_node=False, default=jnp.zeros(1.0))

    def evaluate(self, freqs: ArrayLike) -> Array:
        if jnp.ndim(self.constant) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.constant


@dataclass
class Exp(Kernel):
    r"""
    This kernel, in real space, represents a covariance
    function equal to an exponential decay, given by

    .. math::
        g(r) = \kappa \exp(- r / \xi),

    where :math:`r` is a radial coordinate. The power spectrum
    from such a correlation function (in two-dimensions) is given
    by its Hankel transform pair

    .. math::
        P(k) = \frac{\kappa}{\xi} \frac{1}{(\xi^{-2} + k^2)^{3/2}},

    Attributes
    ----------
    amplitude : `cryojax.core.Float`
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation.
    scale : `cryojax.core.Float`
        The length scale of the kernel, equal to :math:`\xi`
        in the above equation.
    constant : `cryojax.core.Float`
        A constant offset for the kernel, added to the above equation.
    """

    amplitude: Float = field(pytree_node=False, default=jnp.array(1.0))
    scale: Float = field(pytree_node=False, default=jnp.array(1.0))
    constant: Float = field(pytree_node=False, default=jnp.array(0.0))

    def evaluate(self, freqs: ArrayLike) -> Array:
        if self.scale != 0.0:
            k_sqr = jnp.linalg.norm(freqs, axis=-1) ** 2
            scaling = 1.0 / (k_sqr + jnp.divide(1, (self.scale) ** 2)) ** 1.5
            scaling *= jnp.divide(self.amplitude, self.scale)
        else:
            scaling = 0.0
        return scaling + self.constant


@dataclass
class ExpSquared(Kernel):
    r"""
    This kernel represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 4),

    where :math:`k^2` is the norm squared of the
    wave vector. Here, :math:`\beta` has dimensions
    of length squared.

    Attributes
    ----------
    amplitude : `cryojax.core.Float`
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation.
    scale : `cryojax.core.Float`
        The length scale of the kernel, equal to :math:`\beta`
        in the above equation.
    constant : `cryojax.core.Float`
        A constant offset for the kernel, added to the above equation.
    """

    amplitude: Float = field(pytree_node=False, default=jnp.array(1.0))
    scale: Float = field(pytree_node=False, default=jnp.array(1.0))
    constant: Float = field(pytree_node=False, default=jnp.array(0.0))

    def evaluate(self, freqs: ArrayLike) -> Array:
        k_sqr = jnp.linalg.norm(freqs, axis=-1) ** 2
        scaling = self.amplitude * jnp.exp(-0.25 * self.scale * k_sqr)
        return scaling + self.constant


@dataclass
class Empirical(Kernel):
    r"""
    This kernel stores a measured array, rather
    than computing one from a model. The array
    is given a scaling and an offset.

    Attributes
    ----------
    amplitude : `cryojax.core.Float`
        An amplitude scaling for the kernel.
    constant : `cryojax.core.Float`
        A constant offset for the kernel.
    """

    measurement: Array = field(pytree_node=False)

    amplitude: Float = field(pytree_node=False, default=jnp.array(1.0))
    constant: Float = field(pytree_node=False, default=jnp.array(0.0))

    def evaluate(self, freqs: ArrayLike) -> Array:
        """Return the scaled and offset measurement."""
        return self.amplitude * self.measurement + self.constant


@dataclass
class Custom(Kernel):
    """
    A custom kernel class implemented as a callable.

    Attributes
    ----------
    function: `Callable`
        A callable with a signature and behavior that matches
        :func:`Kernel.evaluate`.
    """

    function: Callable[Concatenate[ArrayLike, P], Array]

    def evaluate(
        self, freqs: ArrayLike, *args: P.args, **kwargs: P.kwargs
    ) -> Array:
        return self.function(freqs, *args, **kwargs)

"""
Implementation of a Kernel. Put simply, these are fourier-space 
functions commonly applied to images. The word "kernel" is
borrowed from the theory of gaussian processes, where a kernel
is jargon for a covariance function. These functions are
technically the fourier-space version of any stationary
covariance kernel, but in ``cryojax`` the term kernel is used
in a broader sense.

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

from abc import ABCMeta, abstractmethod
from typing import Any, Union, Callable, Concatenate, ParamSpec, Optional
from functools import partial

import jax.numpy as jnp

from ..core import (
    dataclass,
    field,
    Float,
    Parameter,
    ArrayLike,
    Array,
    CryojaxObject,
)

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

        Parameters
        ----------
        freqs : `ArrayLike`, shape `(..., 2)`
            The wave vectors in the imaging plane, in
            cartesain coordinates.
        """
        pass

    def __call__(self, freqs: ArrayLike, *args: Any, **kwargs: Any) -> Array:
        freqs = jnp.asarray(freqs)
        return self.evaluate(freqs, *args, **kwargs)

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

    kernel1: Kernel = field()
    kernel2: Kernel = field()

    def evaluate(self, freqs: ArrayLike) -> Array:
        return self.kernel1.evaluate(freqs) + self.kernel2.evaluate(freqs)


@dataclass
class Product(Kernel):
    """A helper to represent the product of two kernels"""

    kernel1: Kernel = field()
    kernel2: Kernel = field()

    def evaluate(self, freqs: ArrayLike) -> Array:
        return self.kernel1.evaluate(freqs) * self.kernel2.evaluate(freqs)


@dataclass
class Constant(Kernel):
    """
    This kernel returns a constant.

    Attributes
    ----------
    value : `cryojax.core.Parameter`
        The value of the kernel.
    """

    value: Parameter = field(default=1.0)

    def evaluate(self, freqs: Optional[ArrayLike] = None) -> Array:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant kernel must be a scalar")
        return self.value


@dataclass
class Exp(Kernel):
    r"""
    This kernel, in real space, represents a covariance
    function equal to an exponential decay, given by

    .. math::
        g(r) = \kappa \exp(- r / \xi),

    where :math:`r = \sqrt{x^2 + y^2}` is a radial coordinate.
    The power spectrum from such a correlation function (in two-dimensions)
    is given by its Hankel transform pair

    .. math::
        P(k) = \frac{\kappa}{\xi} \frac{1}{(\xi^{-2} + k^2)^{3/2}}.

    Here, :math:`\xi` has dimensions of length.

    Attributes
    ----------
    amplitude : `cryojax.core.Parameter`
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation. Note that this has dimensions
        of inverse volume.
    scale : `cryojax.core.Parameter`
        The length scale of the kernel, equal to :math:`\xi`
        in the above equation.
    offset : `cryojax.core.Parameter`
        An offset added to the above equation.
    """

    amplitude: Parameter = field(default=0.1)
    scale: Parameter = field(default=1.0)
    offset: Parameter = field(default=0.0)

    def evaluate(self, freqs: ArrayLike) -> Array:
        if self.scale != 0.0:
            k_sqr = jnp.sum(freqs**2, axis=-1)
            scaling = 1.0 / (k_sqr + jnp.divide(1, (self.scale) ** 2)) ** 1.5
            scaling *= jnp.divide(self.amplitude, self.scale)
        else:
            scaling = 0.0
        return scaling + self.offset


@dataclass
class Gaussian(Kernel):
    r"""
    This kernel represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 2),

    where :math:`k^2` is the length of the wave vector.
    Here, :math:`\beta` has dimensions of length squared.

    Attributes
    ----------
    amplitude : `cryojax.core.Parameter`
        The amplitude of the kernel, equal to :math:`\kappa`
        in the above equation.
    b_factor : `cryojax.core.Parameter`
        The length scale of the kernel, equal to :math:`\beta`
        in the above equation.
    offset : `cryojax.core.Parameter`
        An offset added to the above equation.
    """

    amplitude: Parameter = field(default=1.0)
    b_factor: Parameter = field(default=1.0)
    offset: Parameter = field(default=0.0)

    def evaluate(self, freqs: ArrayLike) -> Array:
        k_sqr = jnp.sum(freqs**2, axis=-1)
        scaling = self.amplitude * jnp.exp(-0.5 * self.b_factor * k_sqr)
        return scaling + self.offset


@dataclass
class Empirical(Kernel):
    r"""
    This kernel stores a measured array, rather than
    computing one from a model.

    Attributes
    ----------
    amplitude : `cryojax.core.Parameter`
        An amplitude scaling for the kernel.
    offset : `cryojax.core.Parameter`
        An offset added to the above equation.
    """

    measurement: Array = field(pytree_node=False)

    amplitude: Parameter = field(default=1.0)
    offset: Parameter = field(default=0.0)

    def evaluate(self, freqs: Optional[ArrayLike] = None) -> Array:
        """Return the scaled and offset measurement."""
        return self.amplitude * self.measurement + self.offset


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

    function: Callable[Concatenate[ArrayLike, P], Array] = field(
        pytree_node=False
    )

    def evaluate(
        self, freqs: ArrayLike, *args: P.args, **kwargs: P.kwargs
    ) -> Array:
        return self.function(freqs, *args, **kwargs)

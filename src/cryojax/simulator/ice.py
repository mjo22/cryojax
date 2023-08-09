"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "ExponentialNoiseIce", "EmpiricalIce"]

import jax.numpy as jnp

from abc import ABCMeta, abstractmethod
from typing import Optional

from .noise import GaussianNoise
from ..core import dataclass, field, Serializable, Array, Scalar


@dataclass
class Ice(Serializable, metaclass=ABCMeta):
    """
    Base class for an ice model.
    """


@dataclass
class NullIce(Ice, GaussianNoise):
    """
    A 'null' ice model.
    """

    def sample(self, freqs: Optional[Array] = None) -> Array:
        return 0.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        return 0.0


@dataclass
class EmpiricalIce(Ice, GaussianNoise):
    """
    Ice modeled as gaussian noise with a
    measured power spectrum.

    Attributes
    ----------
    kappa : `cryojax.core.Scalar`
        A scale factor for the variance.
    spectrum : `jax.Array`, shape `(N1, N2)`
        The measured power spectrum.
    """

    spectrum: Array = field(pytree_node=False)

    kappa: Scalar = 1.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Power spectrum measured from a micrograph."""
        return self.kappa * self.spectrum


@dataclass
class ExponentialNoiseIce(Ice, GaussianNoise):
    r"""
    Ice modeled as gaussian noise with a covariance
    matrix equal to an exponential decay, given by

    .. math::
        g(r) = \kappa \exp(- r / \xi),

    where :math:`r` is a radial coordinate. The power spectrum
    from such a correlation function (in two-dimensions) is given
    by its Hankel transform pair

    .. math::
        P(k) = \frac{\kappa}{\xi} \frac{1}{(\xi^{-2} + k^2)^{3/2}},

    Attributes
    ----------
    kappa : `cryojax.core.Scalar`
        The "coupling strength".
    xi : `cryojax.core.Scalar`
        The correlation length. This is measured
        in pixel units, not in physical length.
    """

    kappa: Scalar = 1.0
    xi: Scalar = 1.0

    def variance(self, freqs: Array) -> Array:
        """Power spectrum modeled by a pure exponential."""
        k_norm = jnp.linalg.norm(freqs, axis=-1)
        scaling = 1.0 / (k_norm**2 + jnp.divide(1, (self.xi) ** 2)) ** 1.5
        return jnp.divide(self.kappa, self.xi) * scaling

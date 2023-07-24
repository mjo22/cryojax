"""
Noise models for cryo-EM images.
"""

__all__ = [
    "Noise",
    "NullNoise",
    "GaussianNoise",
    "WhiteNoise",
    "EmpiricalNoise",
    "LorenzianNoise",
]

from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import jax.numpy as jnp
from jax import random

from ..utils import fft
from ..core import field, dataclass, Array, Scalar, Serializable


@dataclass
class Noise(Serializable, metaclass=ABCMeta):
    """
    Base PyTree container for a noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.sample``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    key: Union[Array, random.PRNGKeyArray] = field(pytree_node=False)

    def __post_init__(self):
        # The below is a nasty hack, required for deserialization
        object.__setattr__(self, "key", self.key.astype(jnp.uint32))

    @abstractmethod
    def sample(self, freqs: Array) -> Array:
        """
        Sample a realization of the noise.

        Parameters
        ----------
        freqs : `jax.Array`, shape `(N1, N2, 2)`
            The wave vectors in the imaging plane,
            in pixel units.
        """
        raise NotImplementedError


class GaussianNoise(Noise):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.variance``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    def sample(self, freqs: Array) -> Array:
        spectrum = self.variance(freqs)
        white_noise = fft(random.normal(self.key, shape=freqs.shape[0:-1]))
        return spectrum * white_noise

    @abstractmethod
    def variance(self, freqs: Optional[Array] = None) -> Array:
        """
        The variance tensor of the gaussian. Only diagonal
        variances are supported.
        """
        raise NotImplementedError


@dataclass
class NullNoise(Noise):
    """
    This class can be used as a null noise model.
    """

    def sample(self, freqs: Array) -> Array:
        return 0.0


@dataclass
class WhiteNoise(GaussianNoise):
    """
    Gaussian white noise (flat power spectrum).

    Attributes
    ----------
    alpha : `jax_2dtm.types.Scalar`
        Variance of the white noise.
    """

    alpha: Scalar = 1.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Flat power spectrum."""
        return self.alpha


@dataclass
class EmpiricalNoise(GaussianNoise):
    """
    Gaussian noise with a measured power spectrum.

    Attributes
    ----------
    kappa : `jax_2dtm.types.Scalar`
        A scale factor for the variance.
    sigma : `jax_2dtm.types.Scalar`
        An uncorrelated part of the variance.
    spectrum : `jax.Array`, shape `(N1, N2)`
        The measured power spectrum. Compute this
        with ``jax_2dtm.simulator.compute_whitening_filter``.
        This must match the image shape!
    """

    spectrum: Array = field(pytree_node=False)

    kappa: Scalar = 1.0
    alpha: Scalar = 0.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Power spectrum measured from a micrograph."""
        return self.kappa * self.spectrum + self.alpha


@dataclass
class ExponentialNoise(GaussianNoise):
    r"""
    Gaussian noise with a covariance matrix equal to an exponential
    decay, given by

    .. math::
        g(r) = \kappa \exp(- r / \xi),

    where :math:`r` is a radial coordinate. The power spectrum
    from such a correlation function (in two-dimensions) is given
    by its Hankel transform pair

    .. math::
        P(k) = \frac{\kappa}{\xi} \frac{1}{(\xi^{-2} + k^2)^{3/2}} + \sigma,

    where :math:`\sigma` is an uncorrelated contribution,
    typically described as shot noise.

    Attributes
    ----------
    kappa : `jax_2dtm.types.Scalar`
        The "coupling strength".
    xi : `jax_2dtm.types.Scalar`
        The correlation length. This is measured
        in pixel units, not in physical length.
    alpha : `jax_2dtm.types.Scalar`
        The uncorrelated part of the spectrum.
    """

    kappa: Scalar = 1.0
    xi: Scalar = 1.0
    alpha: Scalar = 1.0

    def variance(self, freqs: Array) -> Array:
        """Power spectrum modeled by a pure exponential, plus a flat contribution."""
        if self.xi == 0.0:
            return self.alpha
        else:
            k_norm = jnp.linalg.norm(freqs, axis=-1)
            scaling = (
                1.0 / (k_norm**2 + jnp.divide(1, (self.xi) ** 2)) ** 1.5
            )
            return jnp.divide(self.kappa, self.xi) * scaling + self.alpha

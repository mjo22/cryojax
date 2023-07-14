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
from ..types import field, dataclass, Array, Scalar
from ..core import Serializable


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
    sigma : `jax_2dtm.types.Scalar`
    """

    sigma: Scalar = 1.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Flat power spectrum."""
        return self.sigma**2


@dataclass
class EmpiricalNoise(GaussianNoise):
    """
    Gaussian noise with a measured power spectrum.

    Attributes
    ----------
    sigma : `jax_2dtm.types.Scalar`
        A scale factor for the variance.
    spectrum : `jax.Array`, shape `(N1, N2)`
        The measured power spectrum. Compute this
        with ``jax_2dtm.simulator.compute_whitening_filter``.
        This must match the image shape!
    """

    spectrum: Array = field(pytree_node=False)

    sigma: Scalar = 1.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Power spectrum measured from a micrograph."""
        return self.sigma * self.spectrum


@dataclass
class LorenzianNoise(GaussianNoise):
    """
    Gaussian noise with a lorenzian power spectrum.

    Attributes
    ----------
    sigma : `jax_2dtm.types.Scalar`
        An uncorrelated part of the spectrum.
    kappa : `jax_2dtm.types.Scalar`
        The "coupling strength".
    xi : `jax_2dtm.types.Scalar`
        The correlation length. This is measured
        in pixel units, not in physical length.
    """

    sigma: Scalar = 1.0
    kappa: Scalar = 1.0
    xi: Scalar = 1.0

    def variance(self, freqs: Array) -> Array:
        """Power spectrum modeled by a lorenzian, plus a flat contribution."""
        k_norm = jnp.linalg.norm(freqs, axis=-1)
        lorenzian = 1.0 / (k_norm**2 + jnp.divide(1, (self.xi) ** 2))
        return self.kappa**2 * lorenzian + self.sigma**2

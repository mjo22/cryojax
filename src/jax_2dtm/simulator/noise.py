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
from typing import Union

import jax.numpy as jnp
from jax import random

from .scattering import ImageConfig
from ..utils import fft
from ..types import field, dataclass, Array, Scalar


@dataclass
class Noise(metaclass=ABCMeta):
    """
    Base PyTree container for a noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.sample``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    key: Union[Array, random.PRNGKeyArray] = field(
        pytree_node=False, default=random.PRNGKey(0)
    )

    @abstractmethod
    def sample(self, freqs: Array, config: ImageConfig) -> Array:
        """
        Sample a realization of the noise.
        """
        raise NotImplementedError


class GaussianNoise(Noise):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.variance``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    def sample(self, freqs: Array, config: ImageConfig) -> Array:
        spectrum = self.variance(freqs, config)
        white_noise = fft(random.normal(self.key, shape=config.shape))
        return spectrum * white_noise

    @abstractmethod
    def variance(self, freqs: Array, config: ImageConfig) -> Array:
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

    def sample(self, freqs: Array, config: ImageConfig) -> Array:
        return 0.0


@dataclass
class WhiteNoise(GaussianNoise):
    """
    Gaussian white noise (flat power spectrum).

    Attributes
    ----------
    alpha : `jax_2dtm.types.Scalar`
    """

    alpha: Scalar = 1.0

    def variance(self, freqs: Array, config: ImageConfig) -> Array:
        """Flat power spectrum."""
        return self.alpha


@dataclass
class EmpiricalNoise(GaussianNoise):
    """
    Gaussian noise with an empirical power spectrum.
    """

    alpha: Scalar = 1.0

    def variance(self, freqs: Array, config: ImageConfig) -> Array:
        """Power spectrum measured from a micrograph."""
        raise NotImplementedError


@dataclass
class LorenzianNoise(GaussianNoise):
    """
    Gaussian noise with a lorenzian power spectrum.
    """

    alpha: Scalar = 1.0
    kappa: Scalar = 1.0
    xi: Scalar = 1.0

    def variance(self, freqs: Array, config: ImageConfig) -> Array:
        """Power spectrum modeled by a lorenzian, plus a flat contribution."""
        k_norm = jnp.linalg.norm(freqs, axis=-1)
        lorenzian = (self.kappa / config.pixel_size**2) / (
            k_norm**2 + jnp.divide(1, (self.xi * config.pixel_size) ** 2)
        )
        return lorenzian + self.alpha

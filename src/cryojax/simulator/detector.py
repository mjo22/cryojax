"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "WhiteNoiseDetector"]

import jax.numpy as jnp

from abc import ABCMeta, abstractmethod
from typing import Optional, Any

from .noise import GaussianNoise, Noise
from ..core import dataclass, Array, Scalar, CryojaxObject


@dataclass
class Detector(CryojaxObject, metaclass=ABCMeta):
    """
    Base class for an electron detector.
    """

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Array:
        """Sample a realization from the detector model."""
        raise NotImplementedError


@dataclass
class NullDetector(Noise, Detector):
    """
    A 'null' detector.
    """

    def sample(self, freqs: Array) -> Array:
        shape = freqs.shape[0:-1]
        return jnp.zeros(shape)


@dataclass
class WhiteNoiseDetector(GaussianNoise, Detector):
    """
    A detector with a gaussian white noise model.

    Attributes
    ----------
    alpha : `cryojax.core.Scalar`
        Variance of the white noise.
    """

    alpha: Scalar = 1.0

    def variance(self, freqs: Optional[Array] = None) -> Array:
        """Flat power spectrum."""
        return self.alpha

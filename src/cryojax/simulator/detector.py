"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "WhiteNoiseDetector"]

from abc import ABCMeta, abstractmethod
from typing import Optional

from .noise import GaussianNoise, Noise
from ..core import dataclass, Serializable, Array, Scalar


@dataclass
class Detector(Serializable, metaclass=ABCMeta):
    """
    Base class for an electron detector.
    """


@dataclass
class NullDetector(Detector, Noise):
    """
    A 'null' detector.
    """

    def sample(self, freqs: Array) -> Array:
        shape = freqs.shape[0:-1]
        return jnp.zeros(shape)


@dataclass
class WhiteNoiseDetector(Detector, GaussianNoise):
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

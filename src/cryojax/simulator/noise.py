"""
Noise models for cryo-EM images.
"""

__all__ = ["Noise", "GaussianNoise"]

from abc import ABCMeta, abstractmethod

from jax import random

from .kernel import Kernel, Constant
from ..utils import fft
from ..core import field, dataclass, Array, ArrayLike, CryojaxObject


@dataclass
class Noise(CryojaxObject, metaclass=ABCMeta):
    """
    Base PyTree container for a noise model.

    When writing subclasses,

        1) Overwrite ``Noise.sample``.
    """

    key: Array = field(pytree_node=False)

    @abstractmethod
    def sample(self, freqs: ArrayLike) -> Array:
        """
        Sample a realization of the noise.

        Parameters
        ----------
        freqs : `jax.Array`, shape `(N1, N2, 2)`, optional
            The wave vectors in the imaging plane.
        """
        raise NotImplementedError


@dataclass
class GaussianNoise(Noise):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``GaussianNoise.variance``.
    """

    variance: Kernel = field(default_factory=Constant)

    def sample(self, freqs: ArrayLike) -> Array:
        spectrum = self.variance(freqs)
        white_noise = fft(random.normal(self.key, shape=freqs.shape[0:-1]))
        return spectrum * white_noise

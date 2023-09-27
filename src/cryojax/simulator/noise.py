"""
Noise models for cryo-EM images.
"""

__all__ = ["Noise", "GaussianNoise"]

from abc import ABCMeta, abstractmethod
from typing import Union
from jaxtyping import Array, PRNGKeyArray

from jax import random

from .kernel import Kernel, Constant
from ..utils import fftn
from ..core import field, Module, ImageCoords, ComplexImage


class Noise(Module, metaclass=ABCMeta):
    """
    Base class for a noise model.

    When writing subclasses,

        1) Overwrite ``Noise.sample``.
    """

    key: Union[Array, PRNGKeyArray] = field(
        static=True, default=random.PRNGKey(seed=0)
    )

    @abstractmethod
    def sample(self, freqs: ImageCoords) -> ComplexImage:
        """
        Sample a realization of the noise.

        Parameters
        ----------
        freqs : The wave vectors in the imaging plane.
        """
        raise NotImplementedError


class GaussianNoise(Noise):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``GaussianNoise.variance``.
    """

    variance: Kernel = field(default_factory=Constant)

    def sample(self, freqs: ImageCoords) -> ComplexImage:
        spectrum = self.variance(freqs)
        white_noise = fftn(random.normal(self.key, shape=freqs.shape[0:-1]))
        return spectrum * white_noise

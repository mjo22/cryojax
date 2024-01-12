"""
Noise models for cryo-EM images.
"""

__all__ = ["Noise", "GaussianNoise"]

from abc import abstractmethod
from typing_extensions import override

import jax.random as jr
from jaxtyping import PRNGKeyArray
from equinox import Module

from .kernel import Kernel, Constant
from ..core import field
from ..typing import ImageCoords, ComplexImage


class Noise(Module):
    """
    Base class for a noise model.

    When writing subclasses,

        1) Overwrite ``Noise.sample``.
    """

    @abstractmethod
    def sample(self, key: PRNGKeyArray, freqs: ImageCoords) -> ComplexImage:
        """
        Sample a realization of the noise.

        Parameters
        ----------
        freqs : The wave vectors in the imaging plane.
        """
        raise NotImplementedError


class GaussianNoise(Noise):
    """
    A gaussian noise model in fourier space.

    To specify the variance of the noise, pass a ``Kernel`` to
    ``GaussianNoise.variance``.
    """

    variance: Kernel = field(default_factory=Constant)

    @override
    def sample(self, key: PRNGKeyArray, freqs: ImageCoords) -> ComplexImage:
        spectrum = self.variance(freqs)
        white_noise = jr.normal(key, shape=freqs.shape[0:-1])
        return spectrum * white_noise

"""
Noise models for cryo-EM images. This is a collection of
abstract base classes to be subclasses by stochastic cryojax
models.
"""

__all__ = ["Noise", "GaussianNoise"]

from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Any, Type

import jax.numpy as jnp
from jax import random

from .kernel import Kernel
from ..utils import fft
from ..core import field, dataclass, Array, ArrayLike, CryojaxObject


@dataclass
class Noise(CryojaxObject, metaclass=ABCMeta):
    """
    Base PyTree container for a noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.sample``.
        2) Use the ``cryojax.core.dataclass`` decorator.
    """

    key: Array = field(pytree_node=False)

    @abstractmethod
    def sample(self, freqs: ArrayLike) -> Array:
        """
        Sample a realization of the noise.

        Parameters
        ----------
        freqs : `jax.Array`, shape `(N1, N2, 2)`
            The wave vectors in the imaging plane,
            in pixel units.
        """
        raise NotImplementedError


@dataclass
class GaussianNoise(Noise):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.variance``.
        2) Use the ``cryojax.core.dataclass`` decorator.
    """

    def sample(self, freqs: ArrayLike) -> Array:
        freqs = jnp.asarray(freqs)
        spectrum = self.variance(freqs)
        white_noise = fft(random.normal(self.key, shape=freqs.shape[0:-1]))
        return spectrum * white_noise

    @abstractmethod
    def variance(self, freqs: Optional[ArrayLike] = None) -> Array:
        """
        The variance tensor of the gaussian. Only diagonal
        variances are supported.
        """
        raise NotImplementedError

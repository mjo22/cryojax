"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override
from typing import Optional

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._stochastic_model import AbstractStochasticModel
from ._config import ImageConfig
from ..image.operators import FourierOperatorLike, Constant
from ..typing import ComplexImage, Image


class AbstractIce(AbstractStochasticModel, strict=True):
    """Base class for an ice model."""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, config: ImageConfig) -> Image:
        """Sample a stochastic realization of the potential due to the ice
        at the exit plane."""
        raise NotImplementedError

    def __call__(
        self,
        key: PRNGKeyArray,
        fourier_potential_at_exit_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Compute the combined potential of the ice and the specimen."""
        # Sample the realization of the potential due to the ice.
        fourier_ice_potential_at_exit_plane = self.sample(key, config)

        return fourier_potential_at_exit_plane + fourier_ice_potential_at_exit_plane


class NullIce(AbstractIce):
    """A "null" ice model."""

    @override
    def sample(self, key: PRNGKeyArray, config: ImageConfig) -> Image:
        return jnp.zeros(config.padded_frequency_grid_in_angstroms.get().shape[0:-1])

    @override
    def __call__(
        self,
        key: PRNGKeyArray,
        potential_at_exit_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        return jnp.zeros(config.padded_frequency_grid_in_angstroms.get().shape[0:-1])


class GaussianIce(AbstractIce, strict=True):
    r"""Ice modeled as gaussian noise.

    **Attributes:**

    `variance` : A function that computes the variance
                 of the ice, modeled as colored gaussian noise.
                 The dimensions of this function are a squared
                 phase contrast.
    """

    variance: FourierOperatorLike

    def __init__(self, variance: FourierOperatorLike):
        self.variance = variance

    @override
    def sample(self, key: PRNGKeyArray, config: ImageConfig) -> ComplexImage:
        """Sample a realization of the ice potential as colored gaussian noise."""
        N_pix = np.prod(config.padded_shape)
        frequency_grid_in_angstroms = config.padded_frequency_grid_in_angstroms.get()
        # Compute standard deviation, scaling up by the variance by the number
        # of pixels to make the realization independent pixel-independent in real-space.
        std = jnp.sqrt(N_pix * self.variance(frequency_grid_in_angstroms))
        ice_potential_at_exit_plane = std * jr.normal(
            key,
            shape=frequency_grid_in_angstroms.shape[0:-1],
            dtype=complex,
        ).at[0, 0].set(0.0)

        return ice_potential_at_exit_plane

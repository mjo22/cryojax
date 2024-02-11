"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override
from equinox import field

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._stochastic_model import AbstractStochasticModel
from ._config import ImageConfig
from ..image.operators import FourierOperatorLike, FourierExp2D
from ..typing import ComplexImage, ImageCoords


class AbstractIce(AbstractStochasticModel, strict=True):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        frequency_grid_in_angstroms: ImageCoords,
    ) -> ComplexImage:
        """Sample a stochastic realization of the potential due to the ice
        at the exit plane."""
        raise NotImplementedError

    def __call__(
        self,
        key: PRNGKeyArray,
        potential_at_exit_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Compute the combined potential of the ice and the specimen, ."""
        ice_potential_at_exit_plane = self.sample(
            key, config.padded_frequency_grid_in_angstroms.get()
        )

        return potential_at_exit_plane + ice_potential_at_exit_plane


class NullIce(AbstractIce):
    """
    A "null" ice model.
    """

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        frequency_grid_in_angstroms: ImageCoords,
    ) -> ComplexImage:
        return jnp.zeros(frequency_grid_in_angstroms.shape[0:-1])

    @override
    def __call__(
        self,
        key: PRNGKeyArray,
        potential_at_exit_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        return potential_at_exit_plane


class GaussianIce(AbstractIce, strict=True):
    r"""
    Ice modeled as gaussian noise.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the ice, modeled as noise. By default,
        ``FourierExp()``.
    """

    variance: FourierOperatorLike = field(default_factory=FourierExp2D)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        frequency_grid_in_angstroms: ImageCoords,
    ) -> ComplexImage:
        """Sample a realization of the ice potential as colored gaussian noise."""
        ice_potential_at_exit_plane = self.variance(
            frequency_grid_in_angstroms
        ) * jr.normal(
            key,
            shape=frequency_grid_in_angstroms.shape[0:-1],
            dtype=complex,
        )
        ice_potential_at_exit_plane = ice_potential_at_exit_plane.at[0, 0].set(
            0.0 + 0.0j
        )
        return ice_potential_at_exit_plane

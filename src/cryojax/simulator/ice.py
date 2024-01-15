"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._stochastic_model import StochasticModel
from .scattering import ScatteringModel
from ..image import FourierOperatorLike, FourierExp
from ..core import field
from ..typing import ComplexImage, ImageCoords


class Ice(StochasticModel):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        frequency_grid_in_angstroms: ImageCoords,
    ) -> ComplexImage:
        """Sample a stochastic realization of the ice at the exit plane."""
        raise NotImplementedError

    def __call__(
        self,
        key: PRNGKeyArray,
        image_at_exit_plane: ComplexImage,
        scattering: ScatteringModel,
    ) -> ComplexImage:
        """Compute a realization of the ice surrounding a specimen."""
        ice_at_exit_plane = self.sample(
            key, scattering.padded_frequency_grid_in_angstroms.get()
        )

        return ice_at_exit_plane


class NullIce(Ice):
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


class GaussianIce(Ice):
    r"""
    Ice modeled as gaussian noise.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the ice, modeled as noise. By default,
        ``FourierExp()``.
    """

    variance: FourierOperatorLike = field(default_factory=FourierExp)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        frequency_grid_in_angstroms: ImageCoords,
    ) -> ComplexImage:
        """Sample from a gaussian noise model, with the variance
        modulated by the CTF."""
        return self.variance(frequency_grid_in_angstroms) * jr.normal(
            key, shape=frequency_grid_in_angstroms.shape[0:-1]
        )

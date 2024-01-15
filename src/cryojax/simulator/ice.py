"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import abstractmethod

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import ClassVar

from ._stochastic_model import StochasticModel
from .optics import Optics
from ..image import FourierOperatorLike, FourierExp
from ..core import field
from ..typing import ComplexImage, Image, RealImage, ImageCoords


class Ice(StochasticModel):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        coords: ImageCoords,
        image: ComplexImage,
        optics: Optics,
    ) -> Image:
        """Sample a realization from the model."""
        raise NotImplementedError


class NullIce(Ice):
    """
    A 'null' ice model.
    """

    is_real: ClassVar[bool] = False

    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        coords: ImageCoords,
        image: ComplexImage,
        optics: Optics,
    ) -> Image:
        return image


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

    is_real: ClassVar[bool] = False

    variance: FourierOperatorLike = field(default_factory=FourierExp)

    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        coords: ImageCoords,
        image: ComplexImage,
        optics: Optics,
    ) -> ComplexImage:
        return image + (
            optics(freqs)
            * self.variance(freqs)
            * jr.normal(key, shape=freqs.shape[0:-1])
        )

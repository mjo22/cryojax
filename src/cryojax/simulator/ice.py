"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import abstractmethod
from typing import Optional

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import Module

from .kernel import Kernel, Exp
from .noise import GaussianNoise
from ..core import field
from ..typing import RealImage, ComplexImage, Image, ImageCoords


class Ice(Module):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[Image] = None,
    ) -> Image:
        """Sample a realization from the ice model."""
        raise NotImplementedError


class NullIce(Ice):
    """
    A 'null' ice model.
    """

    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[ComplexImage] = None,
    ) -> RealImage:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1])


class GaussianIce(GaussianNoise, Ice):
    r"""
    Ice modeled as gaussian noise.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the ice, modeled as noise. By default,
        ``Exp()``.
    """

    variance: Kernel = field(default_factory=Exp)

    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[ComplexImage] = None,
    ) -> ComplexImage:
        return super().sample(key, freqs)

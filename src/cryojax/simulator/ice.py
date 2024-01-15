"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import abstractmethod

import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._stochastic_model import StochasticModel
from .optics import Optics
from ..image import FourierOperatorLike, FourierExp
from ..core import field
from ..typing import ComplexImage, Image, ImageCoords


class Ice(StochasticModel):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        freqs: ImageCoords,
        ctf: Image,
    ) -> ComplexImage:
        """Sample the stochastic part of the model."""
        raise NotImplementedError

    def __call__(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        freqs: ImageCoords,
        coords: ImageCoords,
        optics: Optics,
    ) -> ComplexImage:
        """Pass the image through the ice model."""
        ctf = optics(freqs)
        return self.sample(key, image, freqs, ctf)


class NullIce(Ice):
    """
    A "null" ice model.
    """

    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        freqs: ImageCoords,
        ctf: Image,
    ) -> ComplexImage:
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

    variance: FourierOperatorLike = field(default_factory=FourierExp)

    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        freqs: ImageCoords,
        ctf: Image,
    ) -> ComplexImage:
        """Sample from a gaussian noise model, with the variance
        modulated by the CTF."""
        return image + ctf * self.variance(freqs) * jr.normal(
            key, shape=freqs.shape[0:-1]
        )

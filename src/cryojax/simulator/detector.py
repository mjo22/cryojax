"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "GaussianDetector"]

from abc import abstractmethod
from typing import ClassVar
from typing_extensions import override

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import AbstractClassVar

from ._stochastic_model import StochasticModel
from ..image import FourierOperatorLike, RealOperatorLike, Constant, irfftn
from ..core import field
from ..typing import ComplexImage, ImageCoords, RealImage, Image


class Detector(StochasticModel):
    """
    Base class for an electron detector.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        coords_or_freqs: ImageCoords,
        image: ComplexImage,
    ) -> Image:
        """Sample a realization from the detector."""
        raise NotImplementedError


class NullDetector(Detector):
    """
    A 'null' detector.
    """

    is_real: ClassVar[bool] = False

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        coords_or_freqs: ImageCoords,
        image: ComplexImage,
    ) -> Image:
        return image


class GaussianDetector(Detector):
    """
    A detector with a gaussian noise model. By default,
    this is a white noise model.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the detector noise. By default,
        ``Constant()``.
    """

    is_real: ClassVar[bool] = False

    variance: FourierOperatorLike = field(default_factory=Constant)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        coords_or_freqs: ImageCoords,
        image: ComplexImage,
    ) -> ComplexImage:
        noise = self.variance(coords_or_freqs) * jr.normal(
            key, shape=coords_or_freqs.shape[0:-1], dtype=complex
        )
        return image + noise


class PoissonDetector(Detector):
    """
    A detector with a poisson noise model.

    NOTE: This is untested and very much in a beta version.
    """

    is_real: ClassVar[bool] = True

    dose: RealOperatorLike = field(default_factory=Constant)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        coords_or_freqs: ImageCoords,
        image: ComplexImage,
    ) -> RealImage:
        return jr.poisson(
            key,
            self.dose(coords_or_freqs)
            * irfftn(image, s=coords_or_freqs.shape[0:-1]),
        ).astype(float)

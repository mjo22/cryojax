"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "GaussianDetector"]

from abc import abstractmethod
from typing import ClassVar
from typing_extensions import override

import jax.random as jr
from jaxtyping import PRNGKeyArray
from equinox import AbstractClassVar

from .manager import ImageManager
from ._stochastic_model import StochasticModel
from ..image import (
    FourierOperatorLike,
    RealOperatorLike,
    Constant,
    irfftn,
    rfftn,
)
from ..core import field
from ..typing import ComplexImage, ImageCoords, RealImage


class Detector(StochasticModel):
    """
    Base class for an electron detector.
    """

    is_sample_real: AbstractClassVar[bool]

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        coords_or_freqs: ImageCoords,
    ) -> ComplexImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError

    def __call__(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        manager: ImageManager,
    ) -> ComplexImage:
        """Pass the image through the detector model."""
        if self.is_sample_real:
            coordinate_grid = manager.padded_coordinate_grid_in_angstroms.get()
            return rfftn(
                self.sample(
                    key, irfftn(image, s=manager.padded_shape), coordinate_grid
                )
            )
        else:
            frequency_grid = manager.padded_frequency_grid_in_angstroms.get()
            return self.sample(key, image, frequency_grid)


class NullDetector(Detector):
    """
    A 'null' detector.
    """

    is_sample_real: ClassVar[bool] = False

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        coords_or_freqs: ImageCoords,
    ) -> ComplexImage:
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

    is_sample_real: ClassVar[bool] = False

    variance: FourierOperatorLike = field(default_factory=Constant)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        image: ComplexImage,
        coords_or_freqs: ImageCoords,
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

    is_sample_real: ClassVar[bool] = True

    dose: RealOperatorLike = field(default_factory=Constant)

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        image: RealImage,
        coords_or_freqs: ImageCoords,
    ) -> RealImage:
        return jr.poisson(key, self.dose(coords_or_freqs) * image).astype(
            float
        )

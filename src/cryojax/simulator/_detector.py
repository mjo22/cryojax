"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["AbstractDetector", "NullDetector", "GaussianDetector"]

from abc import abstractmethod
from typing import ClassVar
from typing_extensions import override
from equinox import field

import jax.random as jr
from jaxtyping import PRNGKeyArray
from equinox import AbstractClassVar

from ._manager import ImageManager
from ._stochastic_model import AbstractStochasticModel
from ..image.operators import FourierOperatorLike, Constant
from ..image import irfftn, rfftn
from ..typing import ComplexImage, ImageCoords, RealImage


class AbstractDetector(AbstractStochasticModel, strict=True):
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


class NullDetector(AbstractDetector, strict=True):
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


class GaussianDetector(AbstractDetector, strict=True):
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
        noise = noise.at[0, 0].set(0.0 + 0.0j)
        return image + noise


class PoissonDetector(AbstractDetector, strict=True):
    """
    A detector with a poisson noise model.

    NOTE: This is untested and very much in a beta version.
    """

    is_sample_real: ClassVar[bool] = True

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        image: RealImage,
        coords_or_freqs: ImageCoords,
    ) -> RealImage:
        return jr.poisson(key, image).astype(float)

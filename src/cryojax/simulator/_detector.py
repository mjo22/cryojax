"""
Abstraction of electron detectors in a cryo-EM image.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override
from equinox import AbstractVar, field

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._config import ImageConfig
from ._stochastic_model import AbstractStochasticModel
from ..image.operators import Constant, RealOperatorLike, FourierOperatorLike
from ..image import irfftn, rfftn
from ..typing import ComplexImage, RealImage


class AbstractDetector(AbstractStochasticModel, strict=True):
    """Base class for an electron detector."""

    electrons_per_angstrom_squared: RealOperatorLike
    dqe: FourierOperatorLike

    def __init__(
        self,
        electrons_per_angstrom_squared: RealOperatorLike,
        dqe: Optional[FourierOperatorLike] = None,
    ):
        self.electrons_per_angstrom_squared = electrons_per_angstrom_squared
        self.dqe = dqe or Constant(1.0)

    @abstractmethod
    def sample(self, key: PRNGKeyArray, image: RealImage) -> RealImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError

    def __call__(
        self,
        fourier_squared_wavefunction_at_detector_plane: ComplexImage,
        config: ImageConfig,
        key: Optional[PRNGKeyArray] = None,
    ) -> ComplexImage:
        """Pass the image through the detector model."""
        N1, N2 = config.padded_shape
        coordinate_grid = config.padded_coordinate_grid_in_angstroms.get()
        frequency_grid = config.padded_frequency_grid_in_angstroms.get()
        # Compute the time-integrated electron flux in pixels
        electrons_per_pixel = (
            self.electrons_per_angstrom_squared(coordinate_grid) * config.pixel_size**2
        )
        # ... now the total number of electrons over the entire image
        electrons_per_image = N1 * N2 * electrons_per_pixel
        # Normalize the squared wavefunction to a set of probabilities
        fourier_squared_wavefunction_at_detector_plane /= (
            fourier_squared_wavefunction_at_detector_plane[0, 0]
        )
        # Compute the noiseless signal by applying the DQE to the squared wavefunction
        fourier_signal = fourier_squared_wavefunction_at_detector_plane * self.dqe(
            frequency_grid
        )
        if key is None and isinstance(self.electrons_per_angstrom_squared, Constant):
            # If there is no key given and the dose is constant, apply the dose in fourier space and return
            return electrons_per_image * fourier_signal
        else:
            # ... otherwise, go to real space and apply the dose
            expected_electron_events = electrons_per_image * irfftn(
                fourier_signal, s=config.padded_shape
            )
            if key is None:
                # If there is no key given, go to fourier space and return
                return rfftn(expected_electron_events)
            else:
                # ... otherwise, sample from the detector noise model
                return rfftn(self.sample(key, expected_electron_events))


class NullDetector(AbstractDetector):
    """A null detector model."""

    @override
    def __init__(self):
        self.electrons_per_angstrom_squared = Constant(1.0)
        self.dqe = Constant(1.0)

    @override
    def sample(
        self, key: PRNGKeyArray, expected_electron_events: RealImage
    ) -> RealImage:
        return expected_electron_events

    @override
    def __call__(
        self,
        fourier_squared_wavefunction_at_detector_plane: ComplexImage,
        config: ImageConfig,
        key: Optional[PRNGKeyArray] = None,
    ) -> ComplexImage:
        return fourier_squared_wavefunction_at_detector_plane


class GaussianDetector(AbstractDetector, strict=True):
    """A detector with a gaussian noise model. This is the gaussian limit
    of `PoissonDetector`.
    """

    @override
    def sample(
        self, key: PRNGKeyArray, expected_electron_events: RealImage
    ) -> RealImage:
        return expected_electron_events + jnp.sqrt(
            expected_electron_events
        ) * jr.normal(key, expected_electron_events.shape)


class PoissonDetector(AbstractDetector, strict=True):
    """A detector with a poisson noise model."""

    @override
    def sample(
        self, key: PRNGKeyArray, expected_electron_events: RealImage
    ) -> RealImage:
        return jr.poisson(key, expected_electron_events).astype(float)

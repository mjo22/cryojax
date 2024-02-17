"""
Abstraction of electron detectors in a cryo-EM image.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._dose import ElectronDose
from ._config import ImageConfig
from ._stochastic_model import AbstractStochasticModel
from ..image.operators import (
    Constant,
    FourierOperatorLike,
    AbstractFourierOperator,
)
from ..image import irfftn, rfftn
from ..typing import ComplexImage, RealImage, ImageCoords


class IdealDQE(AbstractFourierOperator, strict=True):
    r"""The model for an ideal DQE.

    See Ruskin et. al. "Quantitative characterization of electron detectors for transmission electron microscopy." (2013)
    for details.
    """

    @override
    def __call__(self, frequency_grid_in_nyquist_units: ImageCoords) -> RealImage:
        """**Arguments:**

        `frequency_grid_in_nyquist_units`: A frequency grid given in units of nyquist.
        """
        # Measure spatial frequency in units of Nyquist
        radial_frequency_grid_in_nyquist_units = jnp.linalg.norm(
            frequency_grid_in_nyquist_units, axis=-1
        )
        return jnp.sinc(radial_frequency_grid_in_nyquist_units / 2) ** 2


class AbstractDetector(AbstractStochasticModel, strict=True):
    """Base class for an electron detector."""

    dqe: FourierOperatorLike

    def __init__(self, dqe: FourierOperatorLike):
        self.dqe = dqe

    @abstractmethod
    def sample(self, key: PRNGKeyArray, image: RealImage) -> RealImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError

    def __call__(
        self,
        fourier_squared_wavefunction_at_detector_plane: ComplexImage,
        dose: ElectronDose,
        config: ImageConfig,
        key: Optional[PRNGKeyArray] = None,
    ) -> ComplexImage:
        """Pass the image through the detector model."""
        N_pix = np.prod(config.padded_shape)
        coordinate_grid_in_angstroms = config.padded_coordinate_grid_in_angstroms.get()
        frequency_grid_in_nyquist_units = config.padded_frequency_grid.get() / 0.5
        # Compute the time-integrated electron flux in pixels
        electrons_per_pixel = (
            dose.electrons_per_angstrom_squared(coordinate_grid_in_angstroms)
            * config.pixel_size**2
        )
        # ... now the total number of electrons over the entire image
        electrons_per_image = N_pix * electrons_per_pixel
        # Normalize the squared wavefunction to a set of probabilities
        fourier_squared_wavefunction_at_detector_plane /= (
            fourier_squared_wavefunction_at_detector_plane[0, 0]
        )
        # Compute the noiseless signal by applying the DQE to the squared wavefunction
        fourier_signal = fourier_squared_wavefunction_at_detector_plane * jnp.sqrt(
            self.dqe(frequency_grid_in_nyquist_units)
        )
        if key is None and isinstance(dose.electrons_per_angstrom_squared, Constant):
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
        dose: ElectronDose,
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

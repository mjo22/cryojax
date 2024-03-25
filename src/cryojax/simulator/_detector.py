"""
Abstraction of electron detectors in a cryo-EM image.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import AbstractVar, field, Module
from jaxtyping import Array, Complex, PRNGKeyArray, Shaped

from ..core import error_if_not_fractional
from ..image import irfftn, rfftn
from ..image.operators import AbstractFourierOperator
from ..typing import ComplexImage, ImageCoords, RealImage, RealNumber
from ._config import ImageConfig
from ._dose import ElectronDose


class AbstractDQE(AbstractFourierOperator, strict=True):
    r"""Base class for a detector DQE."""

    fraction_detected_electrons: AbstractVar[Shaped[RealNumber, "..."]]

    @abstractmethod
    def __call__(
        self,
        frequency_grid_maybe_in_angstroms: ImageCoords,
        *,
        pixel_size: Optional[RealNumber] = None,
    ) -> RealImage | RealNumber:
        """**Arguments:**

        - `frequency_grid_maybe_in_angstroms`: A frequency grid given in units of
                                               nyquist.
        """
        raise NotImplementedError


class NullDQE(AbstractDQE, strict=True):
    r"""A model for a null DQE."""

    fraction_detected_electrons: Shaped[RealNumber, "..."]

    def __init__(self):
        self.fraction_detected_electrons = jnp.asarray(1.0)

    @override
    def __call__(
        self,
        frequency_grid_maybe_in_angstroms: ImageCoords,
        *,
        pixel_size: Optional[RealNumber] = None,
    ) -> RealNumber:
        return jnp.asarray(1.0)


class IdealDQE(AbstractDQE, strict=True):
    r"""The model for an ideal DQE.

    See Ruskin et. al. "Quantitative characterization of electron detectors for
    transmission electron microscopy." (2013) for details.
    """

    fraction_detected_electrons: Shaped[RealNumber, "..."] = field(
        default=1.0, converter=error_if_not_fractional
    )

    @override
    def __call__(
        self,
        frequency_grid_maybe_in_angstroms: ImageCoords,
        *,
        pixel_size: Optional[RealNumber] = None,
    ) -> RealImage:
        if pixel_size is None:
            frequency_grid_in_nyquist_units = frequency_grid_maybe_in_angstroms / 0.5
        else:
            frequency_grid_in_nyquist_units = (
                frequency_grid_maybe_in_angstroms * pixel_size
            ) / 0.5
        return (
            self.fraction_detected_electrons**2
            * jnp.sinc(frequency_grid_in_nyquist_units[..., 0] / 2) ** 2
            * jnp.sinc(frequency_grid_in_nyquist_units[..., 1] / 2) ** 2
        )


class AbstractDetector(Module, strict=True):
    """Base class for an electron detector."""

    dqe: AbstractDQE

    def __init__(self, dqe: AbstractDQE):
        self.dqe = dqe

    @abstractmethod
    def sample(
        self, key: PRNGKeyArray, expected_electron_events: RealImage
    ) -> RealImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError

    def __call__(
        self,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        dose: ElectronDose,
        config: ImageConfig,
        key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Pass the image through the detector model."""
        N_pix = np.prod(config.padded_shape)
        frequency_grid = config.wrapped_padded_frequency_grid.get()
        # Compute the time-integrated electron flux in pixels
        electrons_per_pixel = dose.electrons_per_angstrom_squared * config.pixel_size**2
        # ... now the total number of electrons over the entire image
        electrons_per_image = N_pix * electrons_per_pixel
        # Normalize the squared wavefunction to a set of probabilities
        fourier_squared_wavefunction_at_detector_plane /= (
            fourier_squared_wavefunction_at_detector_plane[0, 0]
        )
        # Compute the noiseless signal by applying the DQE to the squared wavefunction
        fourier_signal = fourier_squared_wavefunction_at_detector_plane * jnp.sqrt(
            self.dqe(frequency_grid)
        )
        # Apply the dose
        fourier_expected_electron_events = electrons_per_image * fourier_signal
        if key is None:
            # If there is no key given, return
            return fourier_expected_electron_events
        else:
            # ... otherwise, go to real space, sample, go back to fourier,
            # and return.
            expected_electron_events = irfftn(
                fourier_expected_electron_events, s=config.padded_shape
            )
            return rfftn(self.sample(key, expected_electron_events))


class NullDetector(AbstractDetector):
    """A null detector model."""

    @override
    def __init__(self):
        self.dqe = NullDQE()

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

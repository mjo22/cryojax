"""
Abstraction of the electron microscope. This includes models
for the optics, electron dose, and detector.
"""

from typing import Optional

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ._config import ImageConfig
from ._detector import AbstractDetector
from .._errors import error_if_not_positive
from ..constants import convert_keV_to_angstroms


class Instrument(Module, strict=True):
    """An abstraction of an electron microscope."""

    voltage_in_kilovolts: Float[Array, ""]
    electrons_per_angstrom_squared: Float[Array, ""]
    detector: Optional[AbstractDetector]

    def __init__(
        self,
        voltage_in_kilovolts: float | Float[Array, ""],
        electrons_per_angstroms_squared: float | Float[Array, ""] = 60.0,
        *,
        detector: Optional[AbstractDetector] = None,
    ):
        """**Arguments:**

        - `voltage_in_kilovolts`: The accelerating voltage of the
                                instrument in kilovolts (kV).
        - `electron_per_angstrom_squared`: The integrated electron flux.
        - `detector`: The model for the detector.
        """
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts)
        )
        self.electrons_per_angstrom_squared = error_if_not_positive(
            jnp.asarray(electrons_per_angstroms_squared)
        )
        self.detector = detector

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        return convert_keV_to_angstroms(self.voltage_in_kilovolts)

    def compute_expected_electron_events(
        self,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the expected electron events from the detector."""
        if self.detector is None:
            raise AttributeError(
                "Tried to call `Instrument.compute_expected_electron_events`, "
                "but the `Instrument`'s detector model is `None`. This "
                "is not allowed!"
            )
        fourier_expected_electron_events = self.detector(
            fourier_squared_wavefunction_at_detector_plane,
            config,
            self.electrons_per_angstrom_squared,
            key=None,
        )

        return fourier_expected_electron_events

    def measure_detector_readout(
        self,
        key: PRNGKeyArray,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Measure the readout from the detector."""
        if self.detector is None:
            raise AttributeError(
                "Tried to call `Instrument.measure_detector_readout`, "
                "but the `Instrument`'s detector model is `None`. This "
                "is not allowed!"
            )
        fourier_detector_readout = self.detector(
            fourier_squared_wavefunction_at_detector_plane,
            config,
            self.electrons_per_angstrom_squared,
            key,
        )

        return fourier_detector_readout

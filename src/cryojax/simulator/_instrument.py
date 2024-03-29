"""
Abstraction of the electron microscope. This includes models
for the optics, electron dose, and detector.
"""

from typing import Optional

import jax.numpy as jnp
from equinox import field, Module
from jaxtyping import Array, Complex, PRNGKeyArray, Shaped

from ..constants import convert_keV_to_angstroms
from ..core import error_if_not_positive
from ..typing import RealNumber
from ._config import ImageConfig
from ._detector import AbstractDetector, NullDetector
from ._dose import ElectronDose
from ._optics import AbstractOptics, NullOptics


class Instrument(Module, strict=True):
    """An abstraction of an electron microscope.

    **Attributes:**

    - `voltage_in_kilovolts`: The accelerating voltage of the
                              instrument in kilovolts (kV).
    - `optics`: The model for the instrument optics.
    - `dose`: The model for the exposure to electrons
              during image formation.
    - `detector` : The model of the detector.
    """

    voltage_in_kilovolts: Shaped[RealNumber, "..."] = field(
        converter=error_if_not_positive
    )
    dose: ElectronDose
    optics: AbstractOptics
    detector: AbstractDetector

    def __init__(
        self,
        voltage_in_kilovolts: float | Shaped[RealNumber, "..."],
        *,
        dose: Optional[ElectronDose] = None,
        optics: Optional[AbstractOptics] = None,
        detector: Optional[AbstractDetector] = None,
    ):
        if (optics is None or dose is None) and isinstance(detector, AbstractDetector):
            raise AttributeError(
                "Cannot set Instrument.detector without passing an AbstractOptics and "
                "an ElectronDose."
            )
        self.voltage_in_kilovolts = jnp.asarray(voltage_in_kilovolts)
        self.optics = optics or NullOptics()
        self.dose = dose or ElectronDose(electrons_per_angstrom_squared=100.0)
        self.detector = detector or NullDetector()

    @property
    def wavelength_in_angstroms(self) -> RealNumber:
        return convert_keV_to_angstroms(self.voltage_in_kilovolts)

    def propagate_to_detector_plane(
        self,
        fourier_phase_at_exit_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
        defocus_offset: RealNumber | float = 0.0,
    ) -> (
        Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
        | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        if isinstance(self.optics, NullOptics):
            raise AttributeError(
                "Tried to call `Instrument.propagate_to_detector_plane`, "
                "but the `Instrument`'s optics model is `NullOptics`. This "
                "is not supported!"
            )
        """Propagate the scattering potential with the optics model."""
        fourier_contrast_at_detector_plane = self.optics(
            fourier_phase_at_exit_plane,
            config,
            self.wavelength_in_angstroms,
            defocus_offset=defocus_offset,
        )

        return fourier_contrast_at_detector_plane

    def compute_fourier_squared_wavefunction(
        self,
        fourier_contrast_at_detector_plane: (
            Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
            | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        ),
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = config.padded_shape
        if isinstance(self.optics, NullOptics):
            raise AttributeError(
                "Tried to call `compute_fourier_squared_wavefunction`, "
                "but the `Instrument`'s optics model is `NullOptics`. This "
                "is not supported!"
            )
        elif self.optics.is_linear:
            # ... compute the squared wavefunction directly from the image contrast
            # as |psi|^2 = 1 + 2C.
            fourier_contrast_at_detector_plane = fourier_contrast_at_detector_plane
            fourier_squared_wavefunction_at_detector_plane = (
                (2 * fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
            )
            return fourier_squared_wavefunction_at_detector_plane
        else:
            raise NotImplementedError(
                "Functionality for AbstractOptics.is_linear = False not supported."
            )

    def compute_expected_electron_events(
        self,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the expected electron events from the detector."""
        if isinstance(self.detector, NullDetector):
            raise AttributeError(
                "Tried to call `Instrument.compute_expected_electron_events`, "
                "but the `Instrument`'s detector model is `NullDetector`. This "
                "is not supported!"
            )
        fourier_expected_electron_events = self.detector(
            fourier_squared_wavefunction_at_detector_plane,
            config,
            self.dose.electrons_per_angstrom_squared,
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
        if isinstance(self.detector, NullDetector):
            raise AttributeError(
                "Tried to call `Instrument.measure_detector_readout`, "
                "but the `Instrument`'s detector model is `NullDetector`. This "
                "is not supported!"
            )
        fourier_detector_readout = self.detector(
            fourier_squared_wavefunction_at_detector_plane,
            config,
            self.dose.electrons_per_angstrom_squared,
            key,
        )

        return fourier_detector_readout

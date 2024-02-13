"""
Abstraction of the electron microscope. This includes models
for the optics, detector, and beam.
"""

from jaxtyping import PRNGKeyArray
from equinox import Module, field

import jax.numpy as jnp

from ..image import rfftn, ifftn
from ._ice import AbstractIce
from ._specimen import AbstractSpecimen
from ._scattering import AbstractScatteringMethod
from ._config import ImageConfig
from ._optics import AbstractOptics, NullOptics
from ._detector import AbstractDetector, NullDetector

from ..typing import ComplexImage, RealImage, Image, Real_


class Instrument(Module, strict=True):
    """An abstraction of an electron microscope.

    **Attributes:**

    `optics`: The model for the instrument optics.

    `detector` : The model of the detector.
    """

    optics: AbstractOptics = field(default_factory=NullOptics)
    detector: AbstractDetector = field(default_factory=NullDetector)

    def __check_init__(self):
        if not isinstance(self.detector, NullDetector) and isinstance(
            self.optics, NullOptics
        ):
            raise AttributeError(
                "Cannot set optics model as NullOptics if the detector model is not NullDetector."
            )

    def scatter_to_exit_plane(
        self, specimen: AbstractSpecimen, scattering: AbstractScatteringMethod
    ) -> ComplexImage:
        """Scatter the specimen potential to the exit plane."""
        # Compute the scattering potential in fourier space
        fourier_potential_at_exit_plane = scattering(specimen)

        return fourier_potential_at_exit_plane

    def scatter_to_exit_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        specimen: AbstractSpecimen,
        scattering: AbstractScatteringMethod,
        solvent: AbstractIce,
    ) -> ComplexImage:
        """Scatter the specimen potential to the exit plane, including
        the potential due to the solvent."""
        # Compute the scattering potential in fourier space
        fourier_potential_at_exit_plane = self.scatter_to_exit_plane(
            specimen, scattering
        )
        # Add potential due to the ice to the specimen potential
        fourier_potential_at_exit_plane_with_solvent = solvent(
            key, fourier_potential_at_exit_plane, scattering.config
        )

        return fourier_potential_at_exit_plane_with_solvent

    def propagate_to_detector_plane(
        self,
        fourier_potential_at_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        """Propagate the scattering potential with the optics model."""
        fourier_contrast_or_wavefunction_at_detector_plane = self.optics(
            fourier_potential_at_exit_plane, config, defocus_offset=defocus_offset
        )

        return fourier_contrast_or_wavefunction_at_detector_plane

    def compute_fourier_squared_wavefunction(
        self,
        fourier_contrast_or_wavefunction_at_detector_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Compute the squared wavefunction at the detector plane, given either the
        contrast or the wavefunction.
        """
        N1, N2 = config.padded_shape
        if isinstance(self.optics, NullOptics):
            # If there is no optics model, assume that the potential is being passed
            # and return unchanged
            fourier_potential_in_exit_plane = (
                fourier_contrast_or_wavefunction_at_detector_plane
            )
            return fourier_potential_in_exit_plane
        elif self.optics.is_linear:
            # ... compute the squared wavefunction directly from the image contrast
            # as C = 1 - 2 |psi|^2 -> |psi|^2 = (1 - C) / 2
            fourier_contrast_at_detector_plane = (
                fourier_contrast_or_wavefunction_at_detector_plane
            )
            fourier_squared_wavefunction_at_detector_plane = (
                (-fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
            ) / 2
            return fourier_squared_wavefunction_at_detector_plane
        else:
            # ... otherwise, take the modulus squared
            fourier_wavefunction_at_detector_plane = (
                fourier_contrast_or_wavefunction_at_detector_plane
            )
            fourier_squared_wavefunction_at_detector_plane = rfftn(
                jnp.abs(ifftn(fourier_wavefunction_at_detector_plane)) ** 2
            )
            raise NotImplementedError(
                "Functionality for AbstractOptics.is_linear = False not supported."
            )

    def measure_detector_readout(
        self,
        key: PRNGKeyArray,
        fourier_squared_wavefunction_at_detector_plane: RealImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Measure the readout from the detector."""
        fourier_detector_readout = self.detector(
            fourier_squared_wavefunction_at_detector_plane, config, key
        )

        return fourier_detector_readout

    def compute_expected_electron_events(
        self,
        fourier_squared_wavefunction_at_detector_plane: ComplexImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Compute the expected electron events from the detector."""
        fourier_expected_electron_events = self.detector(
            fourier_squared_wavefunction_at_detector_plane, config, key=None
        )

        return fourier_expected_electron_events

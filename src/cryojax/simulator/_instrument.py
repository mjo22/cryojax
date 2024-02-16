"""
Abstraction of the electron microscope. This includes models
for the optics and detector.
"""

from typing import Optional, overload
from jaxtyping import PRNGKeyArray
from equinox import Module

import jax.numpy as jnp

from ..image import rfftn, ifftn
from ..image.operators import RealOperatorLike
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

    - `optics`: The model for the instrument optics.

    - `detector` : The model of the detector.
    """

    optics: AbstractOptics
    detector: AbstractDetector

    @overload
    def __init__(self): ...

    @overload
    def __init__(self, optics: AbstractOptics): ...

    @overload
    def __init__(self, optics: AbstractOptics, detector: AbstractDetector): ...

    def __init__(
        self,
        optics: Optional[AbstractOptics] = None,
        detector: Optional[AbstractDetector] = None,
    ):
        if optics is None and isinstance(detector, AbstractDetector):
            raise AttributeError(
                "Cannot set Instrument.detector without passing an optics model."
            )
        self.optics = optics or NullOptics()
        self.detector = detector or NullDetector()

    @property
    def wavelength_in_angstroms(self) -> Real_:
        """The wavelength of the incident electrons."""
        return self.optics.wavelength_in_angstroms

    @property
    def electrons_per_angstrom_squared(self) -> RealOperatorLike:
        """The integrated flux of the incident electrons."""
        return self.detector.electrons_per_angstrom_squared

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
        fourier_specimen_potential_at_exit_plane = self.scatter_to_exit_plane(
            specimen, scattering
        )
        # Get the potential of the ice
        fourier_solvent_potential_at_exit_plane = solvent(
            key,
            scattering.config,
            fourier_specimen_potential_at_exit_plane,
        )
        # Add the specimen and solvent potentials
        fourier_potential_at_exit_plane = (
            fourier_specimen_potential_at_exit_plane
            + fourier_solvent_potential_at_exit_plane
        )

        return fourier_potential_at_exit_plane

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
            # as |psi|^2 = 1 - 2C
            fourier_contrast_at_detector_plane = (
                fourier_contrast_or_wavefunction_at_detector_plane
            )
            fourier_squared_wavefunction_at_detector_plane = (
                (-2 * fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
            )
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

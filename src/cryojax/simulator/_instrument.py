"""
Abstraction of the electron microscope. This includes models
for the optics, detector, and beam.
"""

from jaxtyping import PRNGKeyArray
from equinox import Module, field

from ._ice import AbstractIce
from ._specimen import AbstractSpecimen
from ._scattering import AbstractScatteringMethod
from ._config import ImageConfig
from ._optics import AbstractOptics, NullOptics
from ._detector import AbstractDetector, NullDetector

from ..typing import ComplexImage, RealImage, Image, Real_


class Instrument(Module, strict=True):
    """
    An abstraction of an electron microscope.

    Attributes
    ----------
    optics :
        The model for the contrast transfer function.
    detector :
        The model of the detector.
    """

    optics: AbstractOptics = field(default_factory=NullOptics)
    detector: AbstractDetector = field(default_factory=NullDetector)

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
        get_wavefunction: bool = False,
    ) -> Image:
        """Propagate the scattering potential with the optics model."""
        fourier_wavefunction_or_contrast_at_detector_plane = self.optics(
            fourier_potential_at_exit_plane,
            config,
            defocus_offset=defocus_offset,
            get_wavefunction=get_wavefunction,
        )

        return fourier_wavefunction_or_contrast_at_detector_plane

    def measure_detector_readout(
        self,
        key: PRNGKeyArray,
        fourier_wavefunction_at_detector_plane: RealImage,
        config: ImageConfig,
    ) -> ComplexImage:
        """Measure the readout from the detector."""
        fourier_detector_readout = self.detector(
            fourier_wavefunction_at_detector_plane, config, key
        )

        return fourier_detector_readout

    def measure_detector_electron_events(
        self, fourier_wavefunction_at_detector_plane: ComplexImage, config: ImageConfig
    ) -> ComplexImage:
        """Measure the expected electron events from the detector."""
        fourier_expected_electron_events = self.detector(
            fourier_wavefunction_at_detector_plane, config, key=None
        )

        return fourier_expected_electron_events

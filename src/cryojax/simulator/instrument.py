"""
Abstraction of the electron microscope. This includes models
for the optics, detector, and beam.
"""

__all__ = ["Instrument"]

from jaxtyping import PRNGKeyArray
from equinox import Module

from .ice import Ice
from .specimen import SpecimenBase
from .scattering import ScatteringModel
from .optics import Optics, NullOptics
from .exposure import Exposure, NullExposure
from .detector import Detector, NullDetector

from ..core import field
from ..typing import ComplexImage, Real_


class Instrument(Module):
    """
    An abstraction of an electron microscope.

    Attributes
    ----------
    optics :
        The model for the contrast transfer function.
    exposure :
        The model for the exposure to the electron beam.
    detector :
        The model of the detector.
    """

    optics: Optics = field(default_factory=NullOptics)
    exposure: Exposure = field(default_factory=NullExposure)
    detector: Detector = field(default_factory=NullDetector)

    def scatter_to_exit_plane(
        self, specimen: SpecimenBase, scattering: ScatteringModel
    ) -> ComplexImage:
        """Scatter the specimen to the exit plane"""
        # Draw the electron density at a particular conformation and pose
        density = specimen.get_density()
        # Compute the scattering image in fourier space
        image_at_exit_plane = scattering(density, specimen.pose)
        # Pass image through the electron exposure model
        image_at_exit_plane = self.exposure(
            image_at_exit_plane,
            scattering.padded_frequency_grid_in_angstroms.get(),
            shape_in_real_space=scattering.manager.padded_shape,
        )

        return image_at_exit_plane

    def propagate_to_detector_plane(
        self,
        image_at_exit_plane: ComplexImage,
        scattering: ScatteringModel,
        defocus_offset: Real_ | float = 0.0,
    ) -> ComplexImage:
        """Propagate the image with the optics model"""
        image_at_detector_plane = self.optics(
            image_at_exit_plane,
            scattering.padded_frequency_grid_in_angstroms.get(),
            defocus_offset,
        )

        return image_at_detector_plane

    def propagate_to_detector_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        image_at_exit_plane: ComplexImage,
        solvent: Ice,
        scattering: ScatteringModel,
        defocus_offset: Real_ | float = 0.0,
    ) -> ComplexImage:
        """Propagate the image to the detector plane using the solvent model"""
        # Compute the image of the ice in the exit plane
        ice_at_exit_plane = solvent(key, image_at_exit_plane, scattering)
        # Now, propagate the image of the ice to the detector plane
        ice_at_detector_plane = self.propagate_to_detector_plane(
            ice_at_exit_plane, scattering
        )
        # ... and also the image of the specimen to the detector plane
        image_at_detector_plane = self.propagate_to_detector_plane(
            image_at_exit_plane, scattering, defocus_offset=defocus_offset
        )
        return image_at_detector_plane + ice_at_detector_plane

    def measure_detector_readout(
        self,
        key: PRNGKeyArray,
        image_at_detector_plane: ComplexImage,
        scattering: ScatteringModel,
    ) -> ComplexImage:
        """Measure the detector readout"""
        detector_readout = self.detector(
            key,
            image_at_detector_plane,
            scattering.padded_frequency_grid_in_angstroms.get(),
            scattering.padded_coordinate_grid_in_angstroms.get(),
        )

        return detector_readout

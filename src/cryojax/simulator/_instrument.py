"""
Abstraction of the electron microscope. This includes models
for the optics, detector, and beam.
"""

__all__ = ["Instrument"]

from typing import Any
from jaxtyping import PRNGKeyArray
from equinox import Module, field

from ._ice import AbstractIce
from ._specimen import AbstractSpecimen
from ._scattering import AbstractScatteringMethod
from ._optics import AbstractOptics, NullOptics
from ._exposure import AbstractExposure, NullExposure
from ._detector import AbstractDetector, NullDetector

from ..typing import ComplexImage, Real_


class Instrument(Module, strict=True):
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

    optics: AbstractOptics = field(default_factory=NullOptics)
    exposure: AbstractExposure = field(default_factory=NullExposure)
    detector: AbstractDetector = field(default_factory=NullDetector)

    def scatter_to_exit_plane(
        self,
        specimen: AbstractSpecimen,
        scattering: AbstractScatteringMethod,
        **kwargs: Any,
    ) -> ComplexImage:
        """Scatter the specimen to the exit plane"""
        # Compute the scattering image in fourier space
        image_at_exit_plane = scattering(specimen, **kwargs)
        # Pass image through the electron exposure model
        image_at_exit_plane = self.exposure(
            image_at_exit_plane, scattering.manager
        )

        return image_at_exit_plane

    def propagate_to_detector_plane(
        self,
        image_at_exit_plane: ComplexImage,
        scattering: AbstractScatteringMethod,
        defocus_offset: Real_ | float = 0.0,
        **kwargs: Any,
    ) -> ComplexImage:
        """Propagate the image with the optics model"""
        image_at_detector_plane = self.optics(
            image_at_exit_plane,
            scattering.manager,
            defocus_offset=defocus_offset,
            **kwargs,
        )

        return image_at_detector_plane

    def propagate_to_detector_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        image_at_exit_plane: ComplexImage,
        solvent: AbstractIce,
        scattering: AbstractScatteringMethod,
        defocus_offset: Real_ | float = 0.0,
        **kwargs: Any,
    ) -> ComplexImage:
        """Propagate the image to the detector plane using the solvent model"""
        # Compute the image of the ice in the exit plane
        ice_at_exit_plane = solvent(
            key, image_at_exit_plane, scattering.manager
        )
        # Now, propagate the image of the ice to the detector plane
        ice_at_detector_plane = self.propagate_to_detector_plane(
            ice_at_exit_plane, scattering, **kwargs
        )
        # ... and also the image of the specimen to the detector plane
        image_at_detector_plane = self.propagate_to_detector_plane(
            image_at_exit_plane,
            scattering,
            defocus_offset=defocus_offset,
            **kwargs,
        )
        return image_at_detector_plane + ice_at_detector_plane

    def measure_detector_readout(
        self,
        key: PRNGKeyArray,
        image_at_detector_plane: ComplexImage,
        scattering: AbstractScatteringMethod,
        **kwargs: Any,
    ) -> ComplexImage:
        """Measure the detector readout"""
        detector_readout = self.detector(
            key, image_at_detector_plane, scattering.manager, **kwargs
        )

        return detector_readout

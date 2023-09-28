"""
Image formation models.
"""

from __future__ import annotations

__all__ = [
    "ImagePipeline",
    "ScatteringImage",
    "OpticsImage",
    "DetectorImage",
]

from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import jax.numpy as jnp

from .filter import Filter
from .mask import Mask
from .specimen import Specimen
from .helix import Helix
from .scattering import ScatteringConfig
from .state import PipelineState
from ..utils import fftn, irfftn
from ..core import field, Module
from ..types import RealImage, ComplexImage, Image, Real_


class ImagePipeline(Module, metaclass=ABCMeta):
    """
    Base class for an imaging model.

    Call an ``Image`` or its ``render``, ``sample``,
    or ``log_likelihood`` routines to evaluate the model.

    Attributes
    ----------
    specimen :
        The specimen from which to render images.
    state :
        The state of the model pipeline.
    scattering :
        The image and scattering model configuration.
    filters :
        A list of filters to apply to the image.
    masks :
        A list of masks to apply to the image.
    observed :
        The observed data in real space. This must be the same
        shape as ``scattering.shape``. Note that the user
        should preprocess the observed data before passing it
        to the image, such as applying the ``filters`` and
        ``masks``.
    """

    state: PipelineState = field()
    specimen: Union[Specimen, Helix] = field()
    scattering: ScatteringConfig = field()

    filters: list[Filter] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    observed: Optional[RealImage] = field(default=None)

    @abstractmethod
    def render(self, view: bool = True) -> RealImage:
        """
        Render an image given a parameter set.

        Parameters
        ----------
        view : `bool`
            If ``True``, view the cropped,
            masked, and rescaled image in real
            space. If ``False``, return the image
            at this place in the pipeline.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, view: bool = True) -> RealImage:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the protein signal overlayed
            onto the noise. If ``False``, just return
            the noise given at this place in the pipeline.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self) -> Real_:
        """Evaluate the log-likelihood of the data given a parameter set."""
        raise NotImplementedError

    def __call__(self, view: bool = True) -> Union[RealImage, Real_]:
        """
        Evaluate the model at a parameter set.

        If ``Image.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        if self.observed is None:
            return self.sample(view=view)
        else:
            return self.log_likelihood()

    def view(self, image: Image, real: bool = False) -> RealImage:
        """
        View the image. This function applies
        filters, crops the image, then applies masks.
        """
        # Apply filters
        if real:
            if len(self.filters) > 0:
                image = irfftn(self.filter(fftn(image)))
        else:
            image = irfftn(self.filter(image))
        # View
        image = self.mask(self.scattering.crop(image))
        return image

    def filter(self, image: ComplexImage) -> ComplexImage:
        """Apply filters to image."""
        for filter in self.filters:
            image = filter(image)
        return image

    def mask(self, image: RealImage) -> RealImage:
        """Apply masks to image."""
        for mask in self.masks:
            image = mask(image)
        return image

    @property
    def residuals(self) -> RealImage:
        """Return the residuals between the model and observed data."""
        simulated = self.render()
        residuals = self.observed - simulated
        return residuals


class ScatteringImage(ImagePipeline):
    """
    Compute the scattering pattern in the exit plane,
    with a given image formation model at a given pose.
    """

    def render(self, view: bool = True) -> RealImage:
        """Render the scattered wave in the exit plane."""
        # Compute the image at the exit plane at the given pose
        scattering_image = self.specimen.scatter(
            self.scattering, self.state.pose, exposure=self.state.exposure
        )
        if view:
            scattering_image = self.view(scattering_image)

        return scattering_image

    def sample(self, view: bool = True) -> RealImage:
        """Sample the scattered wave in the exit plane."""
        # Compute the image at the exit plane
        scattering_image = self.render(view=False)
        # Sample a realization of the ice
        ice_image = self.state.ice.scatter(
            self.scattering, resolution=self.specimen.resolution
        )
        # Add the ice to the image
        scattering_image += ice_image
        if view:
            scattering_image = self.view(scattering_image)

        return scattering_image

    def log_likelihood(self) -> Real_:
        return jnp.asarray(0.0)


class OpticsImage(ScatteringImage):
    """
    Compute the image at the detector plane,
    moduated by a CTF.
    """

    def render(self, view: bool = True) -> RealImage:
        """Render the image in the detector plane."""
        # Compute image in detector plane
        optics_image = self.specimen.scatter(
            self.scattering,
            self.state.pose,
            exposure=self.state.exposure,
            optics=self.state.optics,
        )
        if view:
            optics_image = self.view(optics_image)

        return optics_image

    def sample(self, view: bool = True) -> RealImage:
        """Sample the image in the detector plane."""
        # Compute the image at the detector plane
        optics_image = self.render(view=False)
        # Sample a realization of the ice
        ice_image = self.state.ice.scatter(
            self.scattering,
            resolution=self.specimen.resolution,
            optics=self.state.optics,
        )
        # Add the ice to the image
        optics_image += ice_image
        if view:
            optics_image = self.view(optics_image)

        return optics_image


class DetectorImage(OpticsImage):
    """
    Compute the detector readout of the image,
    at a given pixel size.
    """

    def render(self, view: bool = True) -> RealImage:
        # Compute image at detector plane
        optics_image = super().render(view=False)
        # Compute image at detector pixel size
        pixelized_image = self.state.detector.pixelize(
            irfftn(optics_image), resolution=self.specimen.resolution
        )
        if view:
            pixelized_image = self.view(pixelized_image, real=True)

        return pixelized_image

    def sample(self, view: bool = True) -> RealImage:
        """Sample an image from the detector readout."""
        # Determine pixel size
        if self.state.detector.pixel_size is not None:
            pixel_size = self.state.detector.pixel_size
        else:
            pixel_size = self.specimen.resolution
        # Frequencies
        freqs = self.scattering.padded_freqs / pixel_size
        # The specimen image at the detector pixel size
        pixelized_image = self.render(view=False)
        # The ice image at the detector pixel size
        ice_image = self.state.ice.scatter(
            self.scattering,
            resolution=pixel_size,
            optics=self.state.optics,
        )
        ice_image = irfftn(ice_image)
        # Measure the detector readout
        image = pixelized_image + ice_image
        noise = self.state.detector.sample(freqs, image=image)
        detector_readout = image + noise
        if view:
            detector_readout = self.view(detector_readout, real=True)

        return detector_readout

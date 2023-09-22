"""
Image formation models.
"""

from __future__ import annotations

__all__ = [
    "Image",
    "ScatteringImage",
    "OpticsImage",
    "DetectorImage",
]

from abc import ABCMeta, abstractmethod
from typing import Union, Optional

from .filter import Filter
from .mask import Mask
from .specimen import Specimen
from .helix import Helix
from .scattering import ScatteringConfig
from .state import PipelineState
from ..utils import fft, irfft
from ..core import (
    dataclass,
    field,
    ParameterDict,
    Array,
    ArrayLike,
    CryojaxObject,
)


@dataclass
class Image(CryojaxObject, metaclass=ABCMeta):
    """
    Base class for an imaging model. Note that the
    model is a PyTree and is therefore immmutable.

    Use ``Image.update`` to return a new model
    with modified parameters, and call ``Image``
    or its ``render``, ``sample``, or ``log_likelihood``
    routines to evaluate the model.

    Attributes
    ----------
    specimen : `cryojax.simulator.Specimen` or `cryojax.simulator.Helix`
        The specimen from which to render images.
    state : `cryojax.simulator.PipelineState`
        The state of the model pipeline.
    scattering : `cryojax.simulator.ScatteringConfig`
        The image and scattering model configuration.
    filters : `list[Filter]`
        A list of filters to apply to the image.
    masks : `list[Mask]`
        A list of masks to apply to the image.
    observed : `Array`, optional
        The observed data in real space. This must be the same
        shape as ``scattering.shape``. Note that the user
        should preprocess the observed data before passing it
        to the image, such as applying the ``filters`` and
        ``masks``.
    """

    state: PipelineState = field()
    specimen: Union[Specimen, Helix] = field()
    scattering: ScatteringConfig = field(pytree_node=False)

    filters: list[Filter] = field(pytree_node=False, default_factory=list)
    masks: list[Mask] = field(pytree_node=False, default_factory=list)
    observed: Optional[Array] = field(pytree_node=False, default=None)

    @abstractmethod
    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
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
    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
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
    def log_likelihood(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
    ) -> Union[float, Array]:
        """Evaluate the log-likelihood of the data given a parameter set."""
        raise NotImplementedError

    def __call__(
        self,
        **params: ParameterDict,
    ) -> Union[float, Array]:
        """
        Evaluate the model at a parameter set.

        If ``Image.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        state = self.state.update(**params)
        specimen = self.specimen.update(**params)
        if self.observed is None:
            return self.sample(state=state, specimen=specimen)
        else:
            return self.log_likelihood(state=state, specimen=specimen)

    def view(self, image: ArrayLike, real: bool = False) -> Array:
        """
        View the image. This function applies
        filters, crops the image, then applies masks.
        """
        # Apply filters
        if real:
            if len(self.filters) > 0:
                image = irfft(self.filter(fft(image)))
        else:
            image = irfft(self.filter(image))
        # View
        image = self.mask(self.scattering.crop(image))
        return image

    def filter(self, image: ArrayLike) -> Array:
        """Apply filters to image."""
        for filter in self.filters:
            image = filter(image)
        return image

    def mask(self, image: ArrayLike) -> Array:
        """Apply masks to image."""
        for mask in self.masks:
            image = mask(image)
        return image

    def residuals(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
    ):
        """Return the residuals between the model and observed data."""
        state = state or self.state
        specimen = specimen or self.specimen
        simulated = self.render(state=state, specimen=specimen)
        residuals = self.observed - simulated
        return residuals


@dataclass
class ScatteringImage(Image):
    """
    Compute the scattering pattern in the exit plane,
    with a given image formation model at a given pose.
    """

    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        """Render the scattered wave in the exit plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute the image at the exit plane at the given pose
        scattering_image = specimen.scatter(
            scattering, state.pose, exposure=state.exposure
        )
        if view:
            scattering_image = self.view(scattering_image)

        return scattering_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        """Sample the scattered wave in the exit plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute the image at the exit plane
        scattering_image = self.render(
            state=state, specimen=specimen, view=False
        )
        # Sample a realization of the ice
        ice_image = state.ice.scatter(
            scattering, resolution=specimen.resolution
        )
        # Add the ice to the image
        scattering_image += ice_image
        if view:
            scattering_image = self.view(scattering_image)

        return scattering_image

    def log_likelihood(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
    ) -> Union[float, Array]:
        return 0.0


@dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the image at the detector plane,
    moduated by a CTF.
    """

    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        """Render the image in the detector plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute image in detector plane
        optics_image = specimen.scatter(
            scattering,
            state.pose,
            exposure=state.exposure,
            optics=state.optics,
        )
        if view:
            optics_image = self.view(optics_image)

        return optics_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        """Sample the image in the detector plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute the image at the detector plane
        optics_image = self.render(state=state, specimen=specimen, view=False)
        # Sample a realization of the ice
        ice_image = state.ice.scatter(
            scattering, resolution=specimen.resolution, optics=state.optics
        )
        # Add the ice to the image
        optics_image += ice_image
        if view:
            optics_image = self.view(optics_image)

        return optics_image


@dataclass
class DetectorImage(OpticsImage):
    """
    Compute the detector readout of the image
    at its pixel size.
    """

    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute image at detector plane
        optics_image = super().render(
            state=state, specimen=specimen, view=False
        )
        # Compute image at detector pixel size
        pixelized_image = state.detector.pixelize(
            irfft(optics_image), resolution=specimen.resolution
        )
        if view:
            pixelized_image = self.view(pixelized_image, real=True)

        return pixelized_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Union[Specimen, Helix]] = None,
        view: bool = True,
    ) -> Array:
        """Sample an image from the detector readout."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Determine pixel size
        if state.detector.pixel_size is not None:
            pixel_size = state.detector.pixel_size
        else:
            pixel_size = specimen.resolution
        # Frequencies
        freqs = scattering.padded_freqs / pixel_size
        # The specimen image at the detector pixel size
        pixelized_image = self.render(
            state=state, specimen=specimen, view=False
        )
        # The ice image at the detector pixel size
        ice_image = state.ice.scatter(
            scattering, resolution=pixel_size, optics=state.optics
        )
        ice_image = irfft(ice_image)
        # Measure the detector readout
        image = pixelized_image + ice_image
        noise = state.detector.sample(freqs, image=image)
        detector_readout = image + noise
        if view:
            detector_readout = self.view(detector_readout, real=True)

        return detector_readout

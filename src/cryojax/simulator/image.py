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

    Use ``ImageModel.update`` to return a new model
    with modified parameters, and call ``ImageModel``
    or its ``render``, ``sample``, or ``log_likelihood``
    routines to evaluate the model.

    Attributes
    ----------
    specimen : `cryojax.simulator.Specimen`
        The specimen from which to render images.
    state : `cryojax.simulator.PipelineState`
        The state of the model pipeline.
    scattering : `cryojax.simulator.ScatteringConfig`
        The image and scattering model configuration.
    filters : `list[Filter]`
        A list of filters to apply to the image. By default, this is a
        ``LowpassFilter`` with used for antialiasing.
    masks : `list[Mask]`
        A list of masks to apply to the image. By default, there are no
        masks.
    observed : `Array`, optional
        The observed data in real space. This must be the same
        shape as ``scattering.shape``.
    """

    state: PipelineState = field()
    specimen: Specimen = field()
    scattering: ScatteringConfig = field(pytree_node=False)

    filters: list[Filter] = field(pytree_node=False, default_factory=list)
    masks: list[Mask] = field(pytree_node=False, default_factory=list)
    observed: Optional[Array] = field(pytree_node=False, default=None)

    @abstractmethod
    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
        _pixel_size: Optional[float] = None,
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
        _pixel_size : `float`, optional
            The pixel size at which to sample from the noise.
            This is an internal convenience parameter to avoid
            boilerplate and should not be used in the
            API.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
        _pixel_size: Optional[float] = None,
    ) -> Array:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the protein signal overlayed
            onto the noise. If ``False``, just return
            the noise given at this place in the pipeline.
        _pixel_size : `float`, optional
            The pixel size at which to sample from the noise.
            This is an internal convenience parameter to avoid
            boilerplate and should not be used in the API.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
    ) -> Union[float, Array]:
        """Evaluate the log-likelihood of the data given a parameter set."""
        raise NotImplementedError

    def __call__(
        self,
        **params: ParameterDict,
    ) -> Union[float, Array]:
        """
        Evaluate the model at a parameter set.

        If ``ImageModel.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        state = self.state.update(**params)
        specimen = self.specimen.update(**params)
        if self.observed is None:
            return self.sample(state=state, specimen=specimen)
        else:
            return self.log_likelihood(state=state, specimen=specimen)

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
        specimen: Optional[Specimen] = None,
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
        specimen: Optional[Specimen] = None,
        view: bool = True,
    ) -> Array:
        """Render the scattered wave in the exit plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Compute the image at the exit plane at the given pose
        scattering_image = specimen.scatter(scattering, state.pose)
        # Optionally filter, crop, and mask image
        if view:
            scattering_image = self.filter(scattering_image)
            scattering_image = self.mask(
                state.exposure.rescale(
                    scattering.crop(irfft(scattering_image))
                )
            )

        return scattering_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
        _pixel_size: Optional[float] = None,
    ) -> Array:
        """Sample the scattered wave in the exit plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather scattering configuration
        padded_freqs, resolution = (
            scattering.padded_freqs,
            specimen.resolution,
        )
        pixel_size = resolution if _pixel_size is None else _pixel_size
        # Sample from ice distribution
        icy_image = state.ice.sample(padded_freqs / pixel_size)
        if view:
            # Render an image with no ice
            scattering_image = self.render(state=state, specimen=specimen)
            # Apply filters to the ice
            icy_image = self.filter(icy_image)
            # View ice
            icy_image = self.mask(scattering.crop(irfft(icy_image)))
            # Create an icy image in the exit plane
            icy_image += scattering_image

        return icy_image

    def log_likelihood(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
    ) -> Union[float, Array]:
        raise NotImplementedError


@dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the image at the detector plane,
    moduated by a CTF at a given electron dose.
    """

    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
    ) -> Array:
        """Render the image in the detector plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather scattering configuration
        padded_freqs, resolution = (
            scattering.padded_freqs,
            specimen.resolution,
        )
        # Compute scattering in the exit plane.
        scattering_image = super().render(
            state=state, specimen=specimen, view=False
        )
        # Compute and apply CTF
        ctf = state.optics(padded_freqs / resolution, pose=state.pose)
        optics_image = state.optics.apply(ctf, scattering_image)
        # Optionally filter, crop, rescale, and mask image
        if view:
            optics_image = self.filter(optics_image)
            optics_image = self.mask(
                state.exposure.rescale(scattering.crop(irfft(optics_image)))
            )

        return optics_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
        _pixel_size: Optional[float] = None,
    ) -> Array:
        """Sample the image in the detector plane."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather scattering configuration
        padded_freqs, resolution = (
            scattering.padded_freqs,
            specimen.resolution,
        )
        pixel_size = resolution if _pixel_size is None else _pixel_size
        # Sample from ice distribution and apply ctf to it
        ctf = state.optics(padded_freqs / pixel_size, pose=state.pose)
        icy_image = super().sample(state=state, specimen=specimen, view=False)
        icy_image = state.optics.apply(ctf, icy_image)
        if view:
            # Render an image with no ice
            optics_image = self.render(state=state, specimen=specimen)
            # Apply filters to the ice
            icy_image = self.filter(icy_image)
            # View ice
            icy_image = self.mask(scattering.crop(irfft(icy_image)))
            # Create icy image in the detector plane
            icy_image += optics_image

        return icy_image


@dataclass
class DetectorImage(OpticsImage):
    """
    Compute the detector readout of the image
    at its pixel size.
    """

    def render(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
    ) -> Array:
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather scattering configuration
        resolution = specimen.resolution
        # Compute image at detector plane
        optics_image = super().render(
            state=state,
            specimen=specimen,
            view=False,
        )
        # Measure image at detector pixel size
        detector_image = state.detector.measure(
            irfft(optics_image), resolution=resolution
        )
        # Optionally filter, crop, rescale, and mask image
        if view:
            if len(self.filters) > 0:
                detector_image = irfft(self.filter(fft(detector_image)))
            detector_image = self.mask(
                state.exposure.rescale(scattering.crop(detector_image))
            )

        return detector_image

    def sample(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
        view: bool = True,
    ) -> Array:
        """Sample an image from the detector readout."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather image configuration
        padded_freqs, resolution = (
            scattering.padded_freqs,
            specimen.resolution,
        )
        if hasattr(state.detector, "pixel_size"):
            pixel_size = state.detector.pixel_size
        else:
            pixel_size = resolution
        # Sample ice at the detector plane
        ice = super().sample(
            state=state,
            specimen=specimen,
            _pixel_size=pixel_size,
            view=False,
        )
        # Sample from noise distribution of detector
        noise = state.detector.sample(padded_freqs / pixel_size)
        noisy_image = ice + noise
        if view:
            # Render an image from noiseless detector readout
            detector_image = self.render(state=state, specimen=specimen)
            # Apply filters to noise
            noisy_image = self.filter(noisy_image)
            # View noise
            noisy_image = self.mask(scattering.crop(irfft(noisy_image)))
            # Detector readout
            noisy_image += detector_image

        return noisy_image

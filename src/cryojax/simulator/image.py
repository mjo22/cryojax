"""
Image formation models.
"""

from __future__ import annotations

__all__ = ["ImagePipeline"]

from typing import Union, Optional
from functools import cached_property

import equinox as eqx
import jax.tree_util as jtu

from .filter import Filter
from .mask import Mask
from .specimen import Specimen
from .assembly import Assembly
from .scattering import ScatteringConfig
from .instrument import Instrument
from .ice import Ice, NullIce
from ..utils import fftn, irfftn
from ..core import field, Module
from ..typing import RealImage, ComplexImage, Image, Real_


class ImagePipeline(Module):
    """
    Base class for an imaging model.

    Call an ``ImagePipeline`` or its ``render``, ``sample``,
    or ``log_probability`` routines.

    Attributes
    ----------
    specimen :
        The specimen from which to render images.
    scattering :
        The image and scattering model configuration.
    instrument :
        The abstraction of the electron microscope.
    solvent :
        The solvent around the specimen.
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

    specimen: Union[Specimen, Assembly] = field()
    scattering: ScatteringConfig = field()
    instrument: Instrument = field(default_factory=Instrument)
    solvent: Ice = field(default_factory=NullIce)

    filters: list[Filter] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    observed: Optional[RealImage] = field(default=None)

    def render(self, view: bool = True) -> RealImage:
        """
        Render an image of a Specimen.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the cropped, masked,
            and filtered image.
        """
        if isinstance(self.specimen, Specimen):
            image = self._render_specimen()
        elif isinstance(self.specimen, Assembly):
            image = self._render_assembly()
        else:
            raise ValueError(
                "The specimen must be either a Specimen or an Assembly."
            )

        if view:
            image = self.view(image)

        return image

    def sample(self, view: bool = True) -> RealImage:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the cropped, masked,
            and filtered image.
        """
        # Determine pixel size
        if self.instrument.detector.pixel_size is not None:
            pixel_size = self.instrument.detector.pixel_size
        else:
            pixel_size = self.specimen.resolution
        # Frequencies
        freqs = self.scattering.padded_freqs / pixel_size
        # The image of the specimen
        specimen_image = self.render(view=False)
        # The image of the solvent
        ice_image = irfftn(
            self.instrument.optics(freqs) * self.solvent.sample(freqs)
        )
        # Measure the detector readout
        image = specimen_image + ice_image
        noise = self.instrument.detector.sample(freqs, image=image)
        detector_readout = image + noise
        if view:
            detector_readout = self.view(detector_readout)

        return detector_readout

    def log_probability(self) -> Real_:
        """Evaluate the log-probability of the data given a parameter set."""
        raise NotImplementedError

    @cached_property
    def residuals(self) -> RealImage:
        """Return the residuals between the model and observed data."""
        simulated = self.render()
        residuals = self.observed - simulated
        return residuals

    def __call__(self, view: bool = True) -> Union[Image, Real_]:
        """
        Evaluate the model at a parameter set.

        If ``Image.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        if self.observed is None:
            return self.sample(view=view)
        else:
            return self.log_probability()

    def view(self, image: Image, real: bool = True) -> RealImage:
        """
        View the image. This function applies
        filters, crops the image, then applies masks.
        """
        # Apply filters to the image
        if real:
            if len(self.filters) > 0:
                image = irfftn(self.filter(fftn(image)))
        else:
            image = irfftn(self.filter(image))
        # Crop and mask the image
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

    def _render_specimen(self) -> RealImage:
        """Render an image of a Specimen."""
        resolution = self.specimen.resolution
        freqs = self.scattering.padded_freqs / resolution
        # Draw the electron density at a particular conformation and pose
        density = self.specimen.realization
        # Compute the scattering image
        image = self.scattering.scatter(density, resolution=resolution)
        # Apply translation
        image *= self.specimen.pose.shifts(freqs)
        # Compute and apply CTF
        ctf = self.instrument.optics(freqs, pose=self.specimen.pose)
        image = ctf * image
        # Apply the electron exposure model
        scaling, offset = self.instrument.exposure.scaling(
            freqs
        ), self.instrument.exposure.offset(freqs)
        image = scaling * image + offset
        # Measure at the detector pixel size
        image = self.instrument.detector.pixelize(
            irfftn(image), resolution=resolution
        )

        return image

    def _render_assembly(self) -> RealImage:
        """Render an image of an Assembly from its subunits."""
        # Draw the subunits
        subunits = self.specimen.subunits
        # Compute all of the subunit images
        render = lambda subunit: eqx.tree_at(
            lambda m: m.specimen, self, subunit
        )._render_specimen()
        images = jtu.tree_map(
            render, subunits, is_leaf=lambda s: isinstance(s, Specimen)
        )
        # Sum the subunit images together
        image = jtu.tree_reduce(lambda x, y: x + y, images)

        return image

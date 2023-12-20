"""
Image formation models.
"""

from __future__ import annotations

__all__ = ["ImagePipeline"]

from typing import Union, get_args, Optional

import equinox as eqx
import jax.tree_util as jtu
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Shaped

from .filter import Filter
from .mask import Mask
from .specimen import Specimen
from .assembly import Assembly
from .scattering import ScatteringModel
from .manager import ImageManager
from .instrument import Instrument
from .detector import NullDetector
from .ice import Ice, NullIce
from ..utils import fftn, ifftn
from ..core import field, Module
from ..typing import RealImage, Image, Real_


_PRNGKeyArrayLike = Shaped[PRNGKeyArray, "M"]


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
        The scattering model.
    instrument :
        The abstraction of the electron microscope.
    solvent :
        The solvent around the specimen.
    filter :
        A filter to apply to the image.
    mask :
        A mask to apply to the image.

    Properties
    ----------
    manager :
        Exposes the API of the scattering model's image
        manager.
    pixel_size :
        The pixel size of the image.
    """

    specimen: Union[Specimen, Assembly] = field()
    scattering: ScatteringModel = field()
    instrument: Instrument = field(default_factory=Instrument)
    solvent: Ice = field(default_factory=NullIce)

    filter: Optional[Filter] = field(default=None)
    mask: Optional[Mask] = field(default=None)

    @property
    def manager(self) -> ImageManager:
        """The ImageManager"""
        return self.scattering.manager

    @property
    def pixel_size(self) -> Real_:
        """The image pixel size."""
        if self.instrument.detector.pixel_size is not None:
            return self.instrument.detector.pixel_size
        else:
            return self.specimen.resolution

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
                "The specimen must an instance of a Specimen or an Assembly."
            )

        if view:
            image = self._view(image)

        return image

    def sample(
        self, key: Union[PRNGKeyArray, _PRNGKeyArrayLike], view: bool = True
    ) -> RealImage:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        key :
            The random number generator key(s). If
            the ``ImagePipeline`` is configured with
            more than one stochastic model (e.g. a detector
            and solvent model), then this parameter could be
            ``jax.random.split(jax.random.PRNGKey(seed), 2)``.
        view : `bool`, optional
            If ``True``, view the cropped, masked,
            and filtered image.
        """
        # Check PRNGKey
        idx = 0  # Keep track of number of stochastic models
        if isinstance(key, get_args(PRNGKeyArray)):
            key = jnp.expand_dims(key, axis=0)
        # Frequencies
        freqs = self.manager.padded_freqs / self.pixel_size
        # The image of the specimen
        image = self.render(view=False)
        if not isinstance(self.solvent, NullIce):
            # The image of the solvent
            ice_image = ifftn(
                self.instrument.optics(freqs)
                * self.solvent.sample(key[idx], freqs)
            ).real
            image = image + ice_image
            idx += 1
        if not isinstance(self.instrument.detector, NullDetector):
            # Measure the detector readout
            noise = self.instrument.detector.sample(
                key[idx], freqs, image=image
            )
            image = image + noise
            idx += 1
        if view:
            image = self._view(image)

        return image

    def __call__(
        self,
        key: Optional[Union[PRNGKeyArray, _PRNGKeyArrayLike]] = None,
        view: bool = True,
    ) -> Image:
        """
        Sample or render an image.
        """
        if key is None:
            return self.render(view=view)
        else:
            return self.sample(key, view=view)

    def _view(self, image: Image, is_real: bool = True) -> RealImage:
        """
        View the image. This function applies
        filters, crops the image, then applies masks.
        """
        # Apply filters to the image
        if self.filter is not None:
            if is_real:
                image = fftn(image)
            image = ifftn(self.filter(image)).real
        # Crop the image
        image = self.manager.crop(image)
        # Mask the image
        if self.mask is not None:
            image = self.mask(image)
        return image

    def _render_specimen(self) -> RealImage:
        """Render an image of a Specimen."""
        resolution = self.specimen.resolution
        freqs = self.manager.padded_freqs / resolution
        # Draw the electron density at a particular conformation and pose
        density = self.specimen.realization
        # Compute the scattering image in fourier space
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
            ifftn(image).real, resolution=resolution
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

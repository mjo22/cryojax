"""
Image formation models.
"""

from __future__ import annotations

__all__ = ["ImagePipeline"]

from typing import Union, get_args, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, Shaped

from .filter import Filter
from .mask import Mask
from .ensemble import Ensemble
from .assembly import Assembly
from .scattering import ScatteringModel
from .manager import ImageManager
from .instrument import Instrument
from .detector import NullDetector
from .ice import Ice, NullIce
from ..utils import fftn, ifftn
from ..core import field, Module
from ..typing import RealImage, ComplexImage, Image


_PRNGKeyArrayLike = Shaped[PRNGKeyArray, "M"]


class ImagePipeline(Module):
    """
    Base class for an imaging model.

    Call an ``ImagePipeline`` or its ``render``, ``sample``,
    or ``log_probability`` routines.

    Attributes
    ----------
    ensemble :
        The ensemble from which to render images.
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
    """

    ensemble: Union[Ensemble, Assembly] = field()
    scattering: ScatteringModel = field()
    instrument: Instrument = field(default_factory=Instrument)
    solvent: Ice = field(default_factory=NullIce)

    filter: Optional[Filter] = field(default=None)
    mask: Optional[Mask] = field(default=None)

    def render(self, view: bool = True) -> RealImage:
        """
        Render an image of a Specimen.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the cropped, masked,
            and filtered image.
        """
        if isinstance(self.ensemble, Ensemble):
            image = self._render_ensemble()
        elif isinstance(self.ensemble, Assembly):
            image = self._render_assembly()
        else:
            raise ValueError(
                "The ensemble must an instance of an Ensemble or an Assembly."
            )

        if view:
            image = self._filter_crop_mask(image)

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
        freqs = self.scattering.padded_frequency_grid_in_angstroms
        # The image of the specimen drawn from the ensemble
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
            image = self._filter_crop_mask(image)

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

    def _filter_crop_mask(
        self, image: Image, is_real: bool = True
    ) -> RealImage:
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
        image = self.scattering.manager.crop_to_shape(image)
        # Mask the image
        if self.mask is not None:
            image = self.mask(image)
        return image

    def _render_ensemble(self, get_real: bool = True) -> Image:
        """Render an image of a Specimen."""
        freqs = self.scattering.padded_frequency_grid_in_angstroms
        # Draw the electron density at a particular conformation and pose
        density = self.ensemble.realization
        # Compute the scattering image in fourier space
        image = self.scattering(density)
        # Apply translation
        image *= self.ensemble.pose.shifts(freqs)
        # Measure the image with the instrument
        image = self._measure_with_instrument(image, get_real=get_real)

        return image

    def _render_assembly(self, get_real: bool = True) -> Image:
        """Render an image of an Assembly from its subunits."""
        # Get the subunits
        subunits = self.ensemble.subunits
        # Create an ImagePipeline for the subunits
        subunit_pipeline = eqx.tree_at(
            lambda m: m.ensemble,
            self,
            subunits,
        )
        # Setup vmap over poses and conformations
        if self.ensemble.conformation is None:
            where_vmap = lambda m: (m.ensemble.pose,)
            n_vmap = 1
        else:
            where_vmap = lambda m: (m.ensemble.pose, m.ensemble.conformation)
            n_vmap = 2
        to_vmap = eqx.tree_at(
            where_vmap,
            jtu.tree_map(lambda _: False, subunit_pipeline),
            tuple(n_vmap * [True]),
        )
        vmap, novmap = eqx.partition(subunit_pipeline, to_vmap)
        # Compute all subunit images and sum
        compute_stack = jax.vmap(
            lambda vmap, novmap: eqx.combine(vmap, novmap)._render_ensemble(
                get_real=False
            ),
            in_axes=(0, None),
        )
        compute_stack_and_sum = jax.jit(
            lambda vmap, novmap: jnp.sum(
                compute_stack(vmap, novmap),
                axis=0,
            )
        )
        image = compute_stack_and_sum(vmap, novmap)

        return ifftn(image).real if get_real else image

    def _measure_with_instrument(
        self, image: ComplexImage, get_real: bool = True
    ) -> Image:
        """Measure an image with the instrument"""
        instrument = self.instrument
        freqs = self.scattering.padded_frequency_grid_in_angstroms
        # Compute and apply CTF
        ctf = instrument.optics(freqs, pose=self.ensemble.pose)
        image = ctf * image
        # Apply the electron exposure model
        scaling, offset = instrument.exposure.scaling(
            freqs
        ), instrument.exposure.offset(freqs)
        image = scaling * image + offset

        if get_real:
            image = ifftn(image).real

        return image

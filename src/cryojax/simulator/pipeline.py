"""
Image formation models.
"""

from __future__ import annotations

__all__ = ["ImagePipeline", "SuperpositionPipeline"]

from typing import Union, Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import Module

from .specimen import SpecimenBase, Conformation
from .pose import Pose
from .scattering import ScatteringModel
from .instrument import Instrument
from .detector import NullDetector
from .ice import Ice, NullIce
from ..image import rfftn, irfftn
from ..image.operators import Filter, Mask
from ..core import field
from ..typing import ComplexImage, RealImage, Image


class ImagePipeline(Module):
    """
    Base class for an imaging model.

    Call an ``ImagePipeline``'s ``render`` and ``sample``,
    routines.

    Attributes
    ----------
    specimen :
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

    specimen: SpecimenBase = field()
    scattering: ScatteringModel = field()
    instrument: Instrument = field(default_factory=Instrument)
    solvent: Ice = field(default_factory=NullIce)

    filter: Optional[Filter] = field(default=None)
    mask: Optional[Mask] = field(default=None)

    def render(
        self,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """
        Render an image of a Specimen.

        Parameters
        ----------
        view_cropped : `bool`, optional
            If ``True``, view the cropped image.
            If ``view_cropped = False``, ``ImagePipeline.filter``,
            ``ImagePipeline.mask``, and normalization with
            ``normalize = True`` are not applied.
        get_real : `bool`, optional
            If ``True``, return the image in real space.
        normalize : `bool`, optional
            If ``True``, normalize the image to mean zero
            and standard deviation 1.
        """
        freqs = self.scattering.padded_frequency_grid_in_angstroms.get()
        # Draw the electron density at a particular conformation and pose
        density = self.specimen.get_density()
        # Compute the scattering image in fourier space
        image = self.scattering(density)
        # Apply translation
        image *= self.specimen.pose.shifts(freqs)
        # Measure the image with the instrument
        # ... first compute and apply CTF
        ctf = self.instrument.optics(freqs, pose=self.specimen.pose)
        image = ctf * image
        # ... then apply the electron exposure model
        scaling, offset = self.instrument.exposure.scaling(
            freqs
        ), self.instrument.exposure.offset(
            freqs, shape_in_real_space=self.scattering.manager.padded_shape
        )
        image = scaling * image + offset

        return self._get_final_image(
            image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> RealImage:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        key :
            The random number generator key.
        view_cropped : `bool`, optional
            If ``True``, view the cropped image.
            If ``view_cropped = False``, ``ImagePipeline.filter``,
            ``ImagePipeline.mask``, and normalization with
            ``normalize = True`` are not applied.
        get_real : `bool`, optional
            If ``True``, return the image in real space.
        normalize : `bool`, optional
            If ``True``, normalize the image to mean zero
            and standard deviation 1.
        """
        idx = 0  # Keep track of number of stochastic models
        if not isinstance(self.solvent, NullIce) and not isinstance(
            self.instrument.detector, NullDetector
        ):
            keys = jax.random.split(key)
        else:
            keys = jnp.expand_dims(key, axis=0)
        # Frequencies and coordinates
        coords = self.scattering.padded_coordinate_grid_in_angstroms.get()
        freqs = self.scattering.padded_frequency_grid_in_angstroms.get()
        # The image of the specimen drawn from the ensemble
        image = self.render(view_cropped=False, get_real=False)
        if not isinstance(self.solvent, NullIce):
            # The image with the solvent.
            image = self.solvent.sample(
                keys[idx], freqs, coords, image, self.instrument.optics
            )
            image = rfftn(image) if self.solvent.is_real else image
            idx += 1
        if not isinstance(self.instrument.detector, NullDetector):
            # Measure the detector readout
            if self.instrument.detector.is_real:
                image = rfftn(
                    self.instrument.detector.sample(keys[idx], coords, image)
                )
            else:
                image = self.instrument.detector.sample(
                    keys[idx], freqs, image
                )
            idx += 1

        return self._get_final_image(
            image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )

    def __call__(
        self,
        key: Optional[PRNGKeyArray] = None,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """Sample an image with the noise models or render an image
        without them.
        """
        if key is None:
            return self.render(
                view_cropped=view_cropped,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            return self.sample(
                key,
                view_cropped=view_cropped,
                get_real=get_real,
                normalize=normalize,
            )

    def crop_and_apply_operators(
        self,
        image: ComplexImage,
        *,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """
        Return an image postprocessed with filters, cropping, and masking
        in either real or fourier space.
        """
        manager = self.scattering.manager
        # Apply filter
        if self.filter is not None:
            image = self.filter(image)
        # Crop and apply mask
        if self.mask is None and manager.padded_shape == manager.shape:
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if normalize:
                image = manager.normalize_image(
                    image, is_real=False, shape_in_real_space=manager.shape
                )
            return irfftn(image, s=manager.shape) if get_real else image
        else:
            # ... otherwise, inverse transform, crop, mask, and normalize
            image = manager.crop_to_shape(
                irfftn(image, s=manager.padded_shape)
            )
            if self.mask is not None:
                image = self.mask(image)
            if normalize:
                image = manager.normalize_image(image, is_real=True)
            return image if get_real else rfftn(image)

    def _get_final_image(
        self,
        image: ComplexImage,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        manager = self.scattering.manager
        if view_cropped:
            return self.crop_and_apply_operators(
                image,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            return irfftn(image, s=manager.padded_shape) if get_real else image


class SuperpositionPipeline(ImagePipeline):
    """
    Compute an image from a superposition of states in
    ``Ensemble``. This assumes that either ``Ensemble.Pose``
    and/or ``Ensemble.conformation`` has a batch dimension.

    This class can be used to compute a micrograph, where there
    are many specimen in the field of view. Or it can be used to
    compute an image from ``Assembly.subunits``.
    """

    @override
    def render(
        self,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """Render the superposition of states in the Ensemble."""
        # Setup vmap over the Ensemble
        is_vmap = lambda x: isinstance(x, (Pose, Conformation))
        to_vmap = jax.tree_util.tree_map(is_vmap, self, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(self, to_vmap)
        # Compute all images and sum
        compute_image = lambda model: super(type(model), model).render(
            view_cropped=False, get_real=False
        )
        # ... vmap to compute a stack of images to superimpose
        compute_stack = jax.vmap(
            lambda vmap, novmap: compute_image(eqx.combine(vmap, novmap)),
            in_axes=(0, None),
        )
        # ... sum over the stack of images and jit
        compute_stack_and_sum = jax.jit(
            lambda vmap, novmap: jnp.sum(
                compute_stack(vmap, novmap),
                axis=0,
            )
        )
        # ... compute the superposition
        image = compute_stack_and_sum(vmap, novmap)

        return self._get_final_image(
            image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )

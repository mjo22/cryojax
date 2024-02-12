"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import Module, AbstractVar

from ._specimen import AbstractSpecimen, AbstractConformation
from ._pose import AbstractPose
from ._scattering import AbstractScatteringMethod
from ._instrument import Instrument
from ._detector import NullDetector
from ._ice import AbstractIce, NullIce
from ._assembly import AbstractAssembly
from ..image import rfftn, irfftn, normalize_image
from ..image.operators import AbstractFilter, AbstractMask
from ..typing import ComplexImage, Image


class AbstractPipeline(Module, strict=True):
    """
    Base class for an imaging model.

    Call an ``ImagePipeline``'s ``render`` and ``sample``,
    routines.
    """

    scattering: AbstractVar[AbstractScatteringMethod]
    instrument: AbstractVar[Instrument]
    solvent: AbstractVar[AbstractIce]

    filter: AbstractVar[Optional[AbstractFilter]]
    mask: AbstractVar[Optional[AbstractMask]]

    @abstractmethod
    def render(
        self,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """
        Render an image of a Specimen without any stochasticity.

        Namely, do not sample from the ``Ice`` and ``Detector``
        models.

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
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """
        Sample an image from a realization of the ``Ice`` and
        ``Detector`` models.

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
        raise NotImplementedError

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
        config = self.scattering.config
        # Apply filter
        if self.filter is not None:
            image = self.filter(image)
        # Crop and apply mask
        if self.mask is None and config.padded_shape == config.shape:
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if normalize:
                image = normalize_image(
                    image, is_real=False, shape_in_real_space=config.shape
                )
            return irfftn(image, s=config.shape) if get_real else image
        else:
            # ... otherwise, inverse transform, mask, crop, and normalize
            image = irfftn(image, s=config.padded_shape)
            if self.mask is not None:
                image = self.mask(image)
            image = config.crop_to_shape(image)
            if normalize:
                image = normalize_image(image, is_real=True)
            return image if get_real else rfftn(image)

    def _get_final_image(
        self,
        image: ComplexImage,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        config = self.scattering.config
        if view_cropped:
            return self.crop_and_apply_operators(
                image,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            return irfftn(image, s=config.padded_shape) if get_real else image


class ImagePipeline(AbstractPipeline, strict=True):
    """Standard image formation pipeline.

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

    specimen: AbstractSpecimen
    scattering: AbstractScatteringMethod
    instrument: Instrument
    solvent: AbstractIce

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        specimen: AbstractSpecimen,
        scattering: AbstractScatteringMethod,
        instrument: Optional[Instrument] = None,
        solvent: Optional[AbstractIce] = None,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.specimen = specimen
        self.scattering = scattering
        self.instrument = instrument or Instrument()
        self.solvent = solvent or NullIce()
        self.filter = filter
        self.mask = mask

    def render(
        self,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """Render an image of a `Specimen` without any stochasticity."""
        # Scattering the specimen to the exit plane
        fourier_potential_at_exit_plane = self.instrument.scatter_to_exit_plane(
            self.specimen, self.scattering
        )
        # Measure the image at the detector plane
        if not isinstance(self.instrument.detector, NullDetector):
            # If there is a detector, get the wavefunction
            fourier_wavefunction_at_detector_plane = (
                self.instrument.propagate_to_detector_plane(
                    fourier_potential_at_exit_plane,
                    self.scattering.config,
                    defocus_offset=self.specimen.pose.offset_z,
                    get_wavefunction=True,
                )
            )
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.instrument.measure_detector_electron_events(
                    fourier_wavefunction_at_detector_plane, self.scattering.config
                )
            )

            fourier_image = fourier_expected_electron_events
        else:
            # ... otherwise, just get the contrast
            fourier_contrast_at_detector_plane = (
                self.instrument.propagate_to_detector_plane(
                    fourier_potential_at_exit_plane,
                    self.scattering.config,
                    defocus_offset=self.specimen.pose.offset_z,
                    get_wavefunction=False,
                )
            )

            fourier_image = fourier_contrast_at_detector_plane

        return self._get_final_image(
            fourier_image,
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
    ) -> Image:
        """Sample an image from a realization of the ``Ice`` and
        ``Detector`` models."""
        idx = 0  # Keep track of number of stochastic models
        if not isinstance(self.solvent, NullIce) and not isinstance(
            self.instrument.detector, NullDetector
        ):
            keys = jax.random.split(key)
        else:
            keys = jnp.expand_dims(key, axis=0)
        if not isinstance(self.solvent, NullIce):
            # Scatter the specimen to the exit plane with the solvent potential
            fourier_potential_at_exit_plane = (
                self.instrument.scatter_to_exit_plane_with_solvent(
                    keys[idx], self.specimen, self.scattering, self.solvent
                )
            )
            idx += 1
        else:
            # ... otherwise, scatter just the specimen to the exit plane
            fourier_potential_at_exit_plane = self.instrument.scatter_to_exit_plane(
                self.specimen, self.scattering
            )

        # Now, measure the image at the detector plane
        if not isinstance(self.instrument.detector, NullDetector):
            # If there is a detector, get the wavefunction
            fourier_wavefunction_at_detector_plane = (
                self.instrument.propagate_to_detector_plane(
                    fourier_potential_at_exit_plane,
                    self.scattering.config,
                    defocus_offset=self.specimen.pose.offset_z,
                    get_wavefunction=True,
                )
            )
            # ... now measure the expected electron events at the detector
            fourier_detector_readout = self.instrument.measure_detector_readout(
                keys[idx],
                fourier_wavefunction_at_detector_plane,
                self.scattering.config,
            )

            fourier_image = fourier_detector_readout
        else:
            # ... otherwise, just get the contrast
            fourier_contrast_at_detector_plane = (
                self.instrument.propagate_to_detector_plane(
                    fourier_potential_at_exit_plane,
                    self.scattering.config,
                    defocus_offset=self.specimen.pose.offset_z,
                    get_wavefunction=False,
                )
            )

            fourier_image = fourier_contrast_at_detector_plane

        return self._get_final_image(
            fourier_image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )


class AssemblyPipeline(AbstractPipeline, strict=True):
    """Compute an image from a superposition of subunits in
    the ``AbstractAssembly``.

    Attributes
    ----------
    assembly :
        The assembly from which to render images.
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

    assembly: AbstractAssembly
    scattering: AbstractScatteringMethod
    instrument: Instrument
    solvent: AbstractIce

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        assembly: AbstractAssembly,
        scattering: AbstractScatteringMethod,
        instrument: Optional[Instrument] = None,
        solvent: Optional[AbstractIce] = None,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.assembly = assembly
        self.scattering = scattering
        self.instrument = instrument or Instrument()
        self.solvent = solvent or NullIce()
        self.filter = filter
        self.mask = mask

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """Sample the ``AbstractAssembly.subunits`` from the stochastic models."""
        if not isinstance(self.solvent, NullIce):
            raise NotImplementedError(
                "The AssemblyPipeline does not currently support sampling from the solvent model."
            )
        # Get the contrast in the detector plane
        fourier_wavefunction_or_contrast_at_detector_plane = (
            self._compute_subunit_superposition()
        )
        if not isinstance(self.instrument.detector, NullDetector):
            # If there is no detector, the final image is just the weak phase contrast
            fourier_contrast_at_detector_plane = (
                fourier_wavefunction_or_contrast_at_detector_plane
            )
            fourier_image = fourier_contrast_at_detector_plane
        else:
            # Otherwise, measure the detector readout
            fourier_wavefunction_at_detector_plane = (
                fourier_wavefunction_or_contrast_at_detector_plane
            )
            fourier_detector_readout = self.instrument.measure_detector_readout(
                key,
                fourier_wavefunction_at_detector_plane,
                self.scattering.config,
            )
            fourier_image = fourier_detector_readout

        return self._get_final_image(
            fourier_image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )

    @override
    def render(
        self,
        *,
        view_cropped: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> Image:
        """Render the superposition of images from the
        ``AbstractAssembly.subunits``."""
        # Get the wavefunction or fourier contrast in the detector plane
        fourier_wavefunction_or_contrast_at_detector_plane = (
            self._compute_subunit_superposition()
        )
        if not isinstance(self.instrument.detector, NullDetector):
            # If there is no detector, the final image is just the weak phase contrast
            fourier_contrast_at_detector_plane = (
                fourier_wavefunction_or_contrast_at_detector_plane
            )
            fourier_image = fourier_contrast_at_detector_plane
        else:
            # Otherwise, measure the expected electron events
            fourier_wavefunction_at_detector_plane = (
                fourier_wavefunction_or_contrast_at_detector_plane
            )
            fourier_expected_electron_events = (
                self.instrument.measure_detector_electron_events(
                    fourier_wavefunction_at_detector_plane,
                    self.scattering.config,
                )
            )
            fourier_image = fourier_expected_electron_events

        return self._get_final_image(
            fourier_image,
            view_cropped=view_cropped,
            get_real=get_real,
            normalize=normalize,
        )

    def _compute_subunit_superposition(self):
        # Get the assembly subunits
        subunits = self.assembly.subunits
        # Setup vmap over the pose and conformation
        is_vmap = lambda x: isinstance(x, (AbstractPose, AbstractConformation))
        to_vmap = jax.tree_util.tree_map(is_vmap, subunits, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(subunits, to_vmap)
        # Compute all images and sum
        get_wavefunction = not isinstance(self.instrument.detector, NullDetector)
        compute_image = lambda spec, scat, ins: ins.propagate_to_detector_plane(
            ins.scatter_to_exit_plane(spec, scat),
            scat.config,
            defocus_offset=spec.pose.offset_z,
            get_wavefunction=get_wavefunction,
        )
        # ... vmap to compute a stack of images to superimpose
        compute_stack = jax.vmap(
            lambda vmap, novmap, scat, ins: compute_image(
                eqx.combine(vmap, novmap), scat, ins
            ),
            in_axes=(0, None, None, None),
        )
        # ... sum over the stack of images and jit
        compute_stack_and_sum = jax.jit(
            lambda vmap, novmap, scat, ins: jnp.sum(
                compute_stack(vmap, novmap, scat, ins),
                axis=0,
            )
        )
        # ... compute the superposition
        fourier_wavefunction_or_contrast_at_detector_plane = compute_stack_and_sum(
            vmap, novmap, self.scattering, self.instrument
        )
        if get_wavefunction:
            fourier_wavefunction_or_contrast_at_detector_plane.at[0, 0].divide(
                jnp.sqrt(self.assembly.n_subunits)
            )

        return fourier_wavefunction_or_contrast_at_detector_plane

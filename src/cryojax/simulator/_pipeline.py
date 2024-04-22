"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..image import irfftn, normalize_image, rfftn
from ..image.operators import AbstractFilter, AbstractMask
from ._detector import AbstractDetector
from ._instrument_config import InstrumentConfig
from ._scattering_theory import AbstractLinearScatteringTheory, AbstractScatteringTheory


class AbstractImagingPipeline(Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImagingPipeline`'s `render` and `sample`,
    routines.
    """

    config: AbstractVar[InstrumentConfig]
    filter: AbstractVar[Optional[AbstractFilter]]
    mask: AbstractVar[Optional[AbstractMask]]

    @abstractmethod
    def render(
        self,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        """Render an image without any stochasticity.

        **Arguments:**

        - `postprocess`: If `True`, view the cropped, filtered, and masked image.
                          If `postprocess = False`, `ImagePipeline.filter`,
                          `ImagePipeline.mask`, and normalization with
                          `normalize = True` are not applied.
        - `get_real`: If `True`, return the image in real space.
        - `normalize`: If `True`, normalize the image to mean zero
                       and standard deviation 1.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        """Sample an image from a realization of the stochastic models contained
        in the `AbstractImagingPipeline`.

        See `ImagePipeline.render` for documentation of keyword arguments.

        **Arguments:**

        - `key`: The random number generator key.
        """
        raise NotImplementedError

    def postprocess(
        self,
        image: Complex[
            Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"
        ],
        *,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
    ):
        """Return an image postprocessed with filters, cropping, and masking
        in either real or fourier space.
        """
        config = self.config
        if self.mask is None and config.padded_shape == config.shape:
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if self.filter is not None:
                image = self.filter(image)
            if normalize:
                image = normalize_image(
                    image, is_real=False, shape_in_real_space=config.shape
                )
            return irfftn(image, s=config.shape) if get_real else image
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            is_filter_applied = True if self.filter is None else False
            if (
                self.filter is not None
                and self.filter.buffer.shape
                == config.wrapped_padded_frequency_grid_in_pixels.get().shape[0:2]
            ):
                # ... apply the filter here if it is the same size as the padded
                # coordinates
                is_filter_applied = True
                image = self.filter(image)
            image = irfftn(image, s=config.padded_shape)
            image = config.crop_to_shape(image)
            if self.mask is not None:
                image = self.mask(image)
            if is_filter_applied or self.filter is None:
                # ... normalize and return if the filter has already been applied
                if normalize:
                    image = normalize_image(image, is_real=True)
                return image if get_real else rfftn(image)
            else:
                # ... otherwise, apply the filter here, normalize, and return. assume
                # the filter is the same size as the non-padded coordinates
                image = self.filter(rfftn(image))
                if normalize:
                    image = normalize_image(
                        image, is_real=False, shape_in_real_space=config.shape
                    )
                return irfftn(image, s=config.shape) if get_real else image

    def _maybe_postprocess(
        self,
        image: Complex[
            Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"
        ],
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        config = self.config
        if postprocess:
            return self.postprocess(
                image,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            return irfftn(image, s=config.padded_shape) if get_real else image


class ContrastImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns the image contrast from a linear
    scattering theory.

    **Attributes:**

    - `config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory. This must be a linear scattering
                           theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    config: InstrumentConfig
    scattering_theory: AbstractLinearScatteringTheory

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        config: InstrumentConfig,
        scattering_theory: AbstractLinearScatteringTheory,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.config = config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        # Compute the squared wavefunction
        fourier_contrast_at_detector_plane = (
            self.scattering_theory.compute_fourier_contrast_at_detector_plane(
                self.config
            )
        )

        return self._maybe_postprocess(
            fourier_contrast_at_detector_plane,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        # Compute the squared wavefunction
        fourier_contrast_at_detector_plane = (
            self.scattering_theory.compute_fourier_contrast_at_detector_plane(
                self.config, key
            )
        )

        return self._maybe_postprocess(
            fourier_contrast_at_detector_plane,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )


class IntensityImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns an intensity distribution---or in other
    words a squared wavefunction.

    **Attributes:**

    - `config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.config = config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        # Compute the squared wavefunction
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_fourier_squared_wavefunction_at_detector_plane(
                self.config,
            )
        )

        return self._maybe_postprocess(
            fourier_squared_wavefunction_at_detector_plane,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_fourier_squared_wavefunction_at_detector_plane(
                self.config, key
            )
        )

        return self._maybe_postprocess(
            fourier_squared_wavefunction_at_detector_plane,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )


class ElectronCountsImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns electron counts, given a
    model for the detector.

    **Attributes:**

    - `config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `detector`: The electron detector.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.config = config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        # Compute the squared wavefunction
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_fourier_squared_wavefunction_at_detector_plane(self.config)
        )
        # ... now measure the expected electron events at the detector
        fourier_expected_electron_events = (
            self.detector.compute_expected_electron_events(
                fourier_squared_wavefunction_at_detector_plane, self.config
            )
        )

        return self._maybe_postprocess(
            fourier_expected_electron_events,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        *,
        postprocess: bool = True,
        get_real: bool = True,
        normalize: bool = False,
    ) -> (
        Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
        | Float[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim}"]
        | Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
        | Complex[Array, "{self.config.padded_y_dim} {self.config.padded_x_dim//2+1}"]
    ):
        keys = jax.random.split(key)
        # Compute the squared wavefunction
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_fourier_squared_wavefunction_at_detector_plane(
                self.config, keys[0]
            )
        )
        # ... now measure the detector readout
        fourier_detector_readout = self.detector.compute_detector_readout(
            keys[1],
            fourier_squared_wavefunction_at_detector_plane,
            self.config,
        )

        return self._maybe_postprocess(
            fourier_detector_readout,
            postprocess=postprocess,
            get_real=get_real,
            normalize=normalize,
        )

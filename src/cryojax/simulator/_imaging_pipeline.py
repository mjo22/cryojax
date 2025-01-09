"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..image import irfftn, rfftn
from ..image.operators import FilterLike, MaskLike
from ._detector import AbstractDetector
from ._instrument_config import InstrumentConfig
from ._scattering_theory import AbstractScatteringTheory


class AbstractImagingPipeline(Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImagingPipeline`'s `render` routine.
    """

    instrument_config: AbstractVar[InstrumentConfig]
    filter: AbstractVar[Optional[FilterLike]]
    mask: AbstractVar[Optional[MaskLike]]

    @abstractmethod
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        get_cropped: bool = True,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Float[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim}",
        ]
        | Complex[
            Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"
        ]
        | Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ]
    ):
        """Render an image without any stochasticity.

        **Arguments:**

        - `rng_key`:
            The random number generator key. If not passed, render an image
            with no stochasticity.
        - `get_cropped`:
            If `True`, return an image cropped to `InstrumentConfig.shape`.
            Otherwise, return an image at the `InstrumentConfig.padded_shape`.
            If `get_cropped = False`, the `AbstractImagingPipeline.filter`
            and `AbstractImagingPipeline.mask` are not applied, overriding
            the booleans `get_masked` and `get_filtered`.
        - `get_real`:
            If `True`, return the image in real space.
        - `get_masked`:
            If `True`, apply mask stored in `AbstractImagingPipeline.mask`.
        - `get_filtered`:
            If `True`, apply filter stored in `AbstractImagingPipeline.filter`.
        """
        raise NotImplementedError

    def postprocess(
        self,
        image: Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ],
        *,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Complex[
            Array,
            "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}",
        ]
    ):
        """Return an image postprocessed with filters, cropping, and masking
        in either real or fourier space.
        """
        instrument_config = self.instrument_config
        if (
            self.mask is None
            and instrument_config.padded_shape == instrument_config.shape
        ):
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if self.filter is not None:
                image = self.filter(image) if get_filtered else image
            return irfftn(image, s=instrument_config.shape) if get_real else image
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            is_filter_applied = True if self.filter is None else False
            if (
                self.filter is not None
                and self.filter.array.shape
                == instrument_config.padded_frequency_grid_in_pixels.shape[0:2]
            ):
                # ... apply the filter here if it is the same size as the padded
                # coordinates
                is_filter_applied = True
                image = self.filter(image) if get_filtered else image
            image = irfftn(image, s=instrument_config.padded_shape)
            image = instrument_config.crop_to_shape(image)
            if self.mask is not None:
                image = self.mask(image) if get_masked else image
            if is_filter_applied or self.filter is None:
                return image if get_real else rfftn(image)
            else:
                # ... otherwise, apply the filter here and return. assume
                # the filter is the same size as the non-padded coordinates
                image = self.filter(rfftn(image)) if get_filtered else rfftn(image)
                return irfftn(image, s=instrument_config.shape) if get_real else image

    def _maybe_postprocess(
        self,
        image: Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ],
        *,
        get_cropped: bool = True,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Float[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim}",
        ]
        | Complex[
            Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"
        ]
        | Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ]
    ):
        instrument_config = self.instrument_config
        if get_cropped:
            return self.postprocess(
                image, get_real=get_real, get_masked=get_masked, get_filtered=get_filtered
            )
        else:
            return irfftn(image, s=instrument_config.padded_shape) if get_real else image


class ContrastImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns the image contrast from a linear
    scattering theory.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory. This must be a linear scattering
                           theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        get_cropped: bool = True,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Float[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim}",
        ]
        | Complex[
            Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"
        ]
        | Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ]
    ):
        # Compute the squared wavefunction
        fourier_contrast_at_detector_plane = (
            self.scattering_theory.compute_contrast_spectrum_at_detector_plane(
                self.instrument_config, rng_key
            )
        )

        return self._maybe_postprocess(
            fourier_contrast_at_detector_plane,
            get_cropped=get_cropped,
            get_real=get_real,
            get_masked=get_masked,
            get_filtered=get_filtered,
        )


class IntensityImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns an intensity distribution---or in other
    words a squared wavefunction.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        get_cropped: bool = True,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Float[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim}",
        ]
        | Complex[
            Array,
            "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ]
    ):
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_intensity_spectrum_at_detector_plane(
                self.instrument_config, rng_key
            )
        )

        return self._maybe_postprocess(
            fourier_squared_wavefunction_at_detector_plane,
            get_cropped=get_cropped,
            get_real=get_real,
            get_masked=get_masked,
            get_filtered=get_filtered,
        )


class ElectronCountingImagingPipeline(AbstractImagingPipeline, strict=True):
    """An image formation pipeline that returns electron counts, given a
    model for the detector.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `detector`: The electron detector.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        get_cropped: bool = True,
        get_real: bool = True,
        get_masked: bool = True,
        get_filtered: bool = True,
    ) -> (
        Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
        | Float[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim}",
        ]
        | Complex[
            Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"
        ]
        | Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ]
    ):
        if rng_key is None:
            # Compute the squared wavefunction
            theory = self.scattering_theory
            fourier_squared_wavefunction_at_detector_plane = (
                theory.compute_intensity_spectrum_at_detector_plane(
                    self.instrument_config
                )
            )
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_squared_wavefunction_at_detector_plane, self.instrument_config
                )
            )

            return self._maybe_postprocess(
                fourier_expected_electron_events,
                get_cropped=get_cropped,
                get_real=get_real,
                get_masked=get_masked,
                get_filtered=get_filtered,
            )
        else:
            keys = jax.random.split(rng_key)
            # Compute the squared wavefunction
            theory = self.scattering_theory
            fourier_squared_wavefunction_at_detector_plane = (
                theory.compute_intensity_spectrum_at_detector_plane(
                    self.instrument_config, keys[0]
                )
            )
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[1],
                fourier_squared_wavefunction_at_detector_plane,
                self.instrument_config,
            )

            return self._maybe_postprocess(
                fourier_detector_readout,
                get_cropped=get_cropped,
                get_real=get_real,
                get_masked=get_masked,
                get_filtered=get_filtered,
            )

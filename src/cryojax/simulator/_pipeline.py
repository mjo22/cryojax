"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..image import irfftn, normalize_image, rfftn
from ..image.operators import AbstractFilter, AbstractMask
from ._config import ImageConfig
from ._instrument import Instrument
from ._scattering_theory import AbstractScatteringTheory


class AbstractPipeline(Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractPipeline`'s `render` and `sample`,
    routines.
    """

    config: AbstractVar[ImageConfig]
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
        in the `AbstractPipeline`.

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
            if self.mask is not None:
                image = self.mask(image)
            image = config.crop_to_shape(image)
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


class ImagePipeline(AbstractPipeline, strict=True):
    """Standard image formation pipeline.

    **Attributes:**

    - `config`: The image configuration.
    - `scattering_theory`: The scattering theory.
    - `instrument`: The properties of the electron microscope.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    config: ImageConfig
    scattering_theory: AbstractScatteringTheory
    instrument: Instrument

    filter: Optional[AbstractFilter]
    mask: Optional[AbstractMask]

    def __init__(
        self,
        config: ImageConfig,
        scattering_theory: AbstractScatteringTheory,
        instrument: Instrument,
        *,
        filter: Optional[AbstractFilter] = None,
        mask: Optional[AbstractMask] = None,
    ):
        self.config = config
        self.scattering_theory = scattering_theory
        self.instrument = instrument
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
                self.config, self.instrument
            )
        )
        if self.instrument.detector is None:
            return self._maybe_postprocess(
                fourier_squared_wavefunction_at_detector_plane,
                postprocess=postprocess,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.instrument.compute_expected_electron_events(
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
        if self.instrument.detector is not None:
            keys = jax.random.split(key)
        else:
            keys = jnp.expand_dims(key, axis=0)
        # Compute the squared wavefunction
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_fourier_squared_wavefunction_at_detector_plane(
                self.config, self.instrument, keys[0]
            )
        )
        if self.instrument.detector is None:
            return self._maybe_postprocess(
                fourier_squared_wavefunction_at_detector_plane,
                postprocess=postprocess,
                get_real=get_real,
                normalize=normalize,
            )
        else:
            # ... now measure the detector readout
            fourier_detector_readout = self.instrument.measure_detector_readout(
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

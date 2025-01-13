"""
Image formation models simulated from gaussian noise distributions.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar, field
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...image import rfftn
from ...image.operators import Constant, FourierOperatorLike
from ...internal import error_if_not_positive
from ...simulator import AbstractImagingPipeline
from ._base_distribution import AbstractDistribution


class AbstractGaussianDistribution(AbstractDistribution, strict=True):
    r"""An `AbstractDistribution` where images are formed via additive
    gaussian noise.

    Subclasses may compute the likelihood in real or fourier space and
    make different assumptions about the variance / covariance.
    """

    imaging_pipeline: AbstractVar[AbstractImagingPipeline]
    signal_scale_factor: AbstractVar[Float[Array, ""]]

    is_signal_normalized: AbstractVar[bool]

    @override
    def sample(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> (
        Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ]
        | Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ]
    ):
        """Sample from the gaussian noise model."""
        return self.compute_signal(get_real=get_real) + self.compute_noise(
            rng_key, get_real=get_real
        )

    @override
    def compute_signal(
        self, *, get_real: bool = True
    ) -> (
        Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ]
        | Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim}"
            " {self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ]
    ):
        """Render the image formation model."""
        if self.is_signal_normalized:
            simulated_image = self.imaging_pipeline.render(
                get_real=True, get_masked=False
            )
            if self.imaging_pipeline.mask is None:
                mean, std = jnp.mean(simulated_image), jnp.std(simulated_image)
                simulated_image = (simulated_image - mean) / std
            else:
                is_signal = self.imaging_pipeline.mask.array == 1.0
                mean, std = (
                    jnp.mean(simulated_image, where=is_signal),
                    jnp.std(simulated_image, where=is_signal),
                )
                simulated_image = self.imaging_pipeline.mask(
                    self.signal_scale_factor * (simulated_image - mean) / std
                )
            return simulated_image if get_real else rfftn(simulated_image)
        else:
            return self.signal_scale_factor * self.imaging_pipeline.render(
                get_real=get_real
            )

    @abstractmethod
    def compute_noise(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> (
        Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ]
        | Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ]
    ):
        """Draw a realization from the gaussian noise model and return either in
        real or fourier space.
        """
        raise NotImplementedError


class IndependentGaussianPixels(AbstractGaussianDistribution, strict=True):
    r"""A gaussian noise model, where each pixel is independently drawn from
    a zero-mean gaussian of fixed variance (white noise).

    This computes the likelihood in real space, where the variance is a
    constant value across all pixels.
    """

    imaging_pipeline: AbstractImagingPipeline
    variance: Float[Array, ""]
    signal_scale_factor: Float[Array, ""]

    is_signal_normalized: bool = field(static=True)

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        variance: float | Float[Array, ""] = 1.0,
        signal_scale_factor: float | Float[Array, ""] = 1.0,
        is_signal_normalized: bool = False,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        - `variance`: The variance of each pixel.
        - `signal_scale_factor`: A scale factor for the underlying signal simulated from `imaging_pipeline`.
        - `is_signal_normalized`:
            Whether or not the signal is normalized before applying the `signal_scale_factor`.
            If an `AbstractMask` is given to `imaging_pipeline.mask`, the signal is normalized
            within the region where the mask is equal to `1`.
        """  # noqa: E501
        self.imaging_pipeline = imaging_pipeline
        self.variance = error_if_not_positive(variance)
        self.signal_scale_factor = error_if_not_positive(signal_scale_factor)
        self.is_signal_normalized = is_signal_normalized

    @override
    def compute_noise(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> (
        Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ]
        | Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ]
    ):
        pipeline = self.imaging_pipeline
        n_pixels = pipeline.instrument_config.padded_n_pixels
        freqs = pipeline.instrument_config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance)
        noise = pipeline.postprocess(
            std
            * jr.normal(rng_key, shape=freqs.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            get_real=get_real,
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ],
    ) -> Float[Array, ""]:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        - `observed` : The observed data in real space.
        """
        variance = self.variance
        # Create simulated data
        simulated = self.compute_signal(get_real=True)
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_pixel = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each pixel.
        log_likelihood_per_pixel = -1.0 * (
            squared_standard_normal_per_pixel - jnp.log(2 * jnp.pi * variance) / 2
        )
        # Compute log-likelihood, summing over pixels
        log_likelihood = jnp.sum(log_likelihood_per_pixel)

        return log_likelihood


class IndependentGaussianFourierModes(AbstractGaussianDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    imaging_pipeline: AbstractImagingPipeline
    variance_function: FourierOperatorLike
    signal_scale_factor: Float[Array, ""]

    is_signal_normalized: bool = field(static=True)

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        variance_function: Optional[FourierOperatorLike] = None,
        signal_scale_factor: float | Float[Array, ""] = 1.0,
        is_signal_normalized: bool = False,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        - `variance_function`: The variance of each fourier mode. By default,
                               `cryojax.image.operators.Constant(1.0)`.
        - `signal_scale_factor`: A scale factor for the underlying signal simulated from `imaging_pipeline`.
        - `is_signal_normalized`:
            Whether or not the signal is normalized before applying the `signal_scale_factor`.
            If an `AbstractMask` is given to `imaging_pipeline.mask`, the signal is normalized
            within the region where the mask is equal to `1`.
        """  # noqa: E501
        self.imaging_pipeline = imaging_pipeline
        self.variance_function = variance_function or Constant(1.0)
        self.signal_scale_factor = error_if_not_positive(jnp.asarray(signal_scale_factor))
        self.is_signal_normalized = is_signal_normalized

    def compute_noise(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> (
        Float[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim}",
        ]
        | Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ]
    ):
        pipeline = self.imaging_pipeline
        n_pixels = pipeline.instrument_config.padded_n_pixels
        freqs = pipeline.instrument_config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance_function(freqs))
        noise = pipeline.postprocess(
            std
            * jr.normal(rng_key, shape=freqs.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            get_real=get_real,
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",
        ],
    ) -> Float[Array, ""]:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        - `observed` : The observed data in fourier space.
        """
        pipeline = self.imaging_pipeline
        n_pixels = pipeline.instrument_config.n_pixels
        freqs = pipeline.instrument_config.frequency_grid_in_angstroms
        # Compute the variance and scale up to be independent of the number of pixels
        variance = n_pixels * self.variance_function(freqs)
        # Create simulated data
        simulated = self.compute_signal(get_real=False)
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_mode = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each fourier mode.
        log_likelihood_per_mode = (
            squared_standard_normal_per_mode - jnp.log(2 * jnp.pi * variance) / 2
        )
        # Compute log-likelihood, throwing away the zero mode. Need to take care
        # to compute the loss function in fourier space for a real-valued function.
        log_likelihood = (
            -1.0
            * (
                jnp.sum(log_likelihood_per_mode[1:, 0])
                + 2 * jnp.sum(log_likelihood_per_mode[:, 1:])
            )
            / n_pixels
        )

        return log_likelihood

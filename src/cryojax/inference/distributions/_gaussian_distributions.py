"""
Image formation models simulated from gaussian noise distributions.
"""

from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import field
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..._errors import error_if_not_positive
from ...image import normalize_image, rfftn
from ...image.operators import Constant, FourierOperatorLike
from ...simulator import AbstractImagingPipeline
from ._base_distribution import AbstractDistribution


class IndependentGaussianFourierModes(AbstractDistribution, strict=True):
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
        signal_scale_factor: Optional[float | Float[Array, ""]] = None,
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
        if signal_scale_factor is None:
            signal_scale_factor = jnp.sqrt(
                jnp.asarray(imaging_pipeline.instrument_config.n_pixels, dtype=float)
            )
        self.signal_scale_factor = error_if_not_positive(jnp.asarray(signal_scale_factor))
        self.is_signal_normalized = is_signal_normalized

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
            simulated_image = self.imaging_pipeline.render(get_real=True)
            where = (
                None
                if self.imaging_pipeline.mask is None
                else self.imaging_pipeline.mask.array == 1.0
            )
            normalized_simulated_image = normalize_image(simulated_image, where=where)
            return (
                self.signal_scale_factor * normalized_simulated_image
                if get_real
                else self.signal_scale_factor * rfftn(normalized_simulated_image)
            )
        else:
            real_or_fourier_simulated_image = self.imaging_pipeline.render(
                get_real=get_real
            )
            return self.signal_scale_factor * real_or_fourier_simulated_image

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

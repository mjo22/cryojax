"""
Image formation models simulated from gaussian noise distributions.
"""

from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..._errors import error_if_not_positive
from ...image import rescale_image
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

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        variance_function: Optional[FourierOperatorLike] = None,
        signal_scale_factor: Optional[float | Float[Array, ""]] = None,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        - `variance_function`: The variance of each fourier mode. By default,
                               `cryojax.image.operators.Constant(1.0)`.
        - `signal_scale_factor`: A scale factor for the standard deviation of the
                                 underlying signal simulated from `imaging_pipeline`.
                                 The standard deviation of the signal is rescaled to be
                                 equal to `signal_scale_factor / jnp.sqrt(n_pixels)`,
                                 where the inverse square root of `n_pixels` is included
                                 so that the scale of the signal does not depend on the
                                 number of pixels. As a result, a good starting value for
                                 `signal_scale_factor` should be on the order of the
                                 extent of the object in pixels. By default,
                                 `signal_scale_factor = sqrt(imaging_pipeline.instrument_config.n_pixels)`.
        """  # noqa: E501
        self.imaging_pipeline = imaging_pipeline
        self.variance_function = variance_function or Constant(1.0)
        if signal_scale_factor is None:
            signal_scale_factor = jnp.sqrt(
                jnp.asarray(imaging_pipeline.instrument_config.n_pixels, dtype=float)
            )
        self.signal_scale_factor = error_if_not_positive(jnp.asarray(signal_scale_factor))

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
        n_pixels = self.imaging_pipeline.instrument_config.n_pixels
        shape = self.imaging_pipeline.instrument_config.shape
        simulated_image = self.imaging_pipeline.render(get_real=get_real)
        return rescale_image(
            simulated_image,
            std=self.signal_scale_factor / jnp.sqrt(n_pixels),
            mean=0.0,
            is_real=get_real,
            shape_in_real_space=shape,
        )

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
        pipeline = self.imaging_pipeline
        freqs = pipeline.instrument_config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        padded_n_pixels = pipeline.instrument_config.padded_n_pixels
        std = jnp.sqrt(padded_n_pixels * self.variance_function(freqs))
        noise = pipeline.postprocess(
            std
            * jr.normal(rng_key, shape=freqs.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            get_real=get_real,
        )
        image = self.compute_signal(get_real=get_real)
        return image + noise

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
        # Compute the log-likelihood for each fourier mode. Divide by the
        # number of pixels so that the likelihood is a sum over pixels in
        # real space (parseval's theorem)
        log_likelihood_per_mode = (
            squared_standard_normal_per_mode - jnp.log(2 * jnp.pi * variance) / 2
        ) / n_pixels
        # Compute log-likelihood, throwing away the zero mode. Need to take care
        # to compute the loss function in fourier space for a real-valued function.
        log_likelihood = -1.0 * (
            jnp.sum(log_likelihood_per_mode[1:, 0])
            + 2 * jnp.sum(log_likelihood_per_mode[:, 1:])
        )

        return log_likelihood

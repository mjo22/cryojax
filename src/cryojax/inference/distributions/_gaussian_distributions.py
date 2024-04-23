"""
Image formation models simulated from gaussian noise distributions.
"""

from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import field
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..._errors import error_if_not_positive
from ...image.operators import Constant, FourierOperatorLike
from ...simulator import AbstractImagingPipeline
from ._base_distribution import AbstractDistribution


class IndependentGaussianFourierModes(AbstractDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    imaging_pipeline: AbstractImagingPipeline
    variance: FourierOperatorLike
    contrast_scale: Float[Array, ""] = field(converter=error_if_not_positive)

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        variance: Optional[FourierOperatorLike] = None,
        contrast_scale: float | Float[Array, ""] = 1.0,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        - `variance`: The variance of each fourier mode. By default,
                      `cryojax.image.operators.Constant(1.0)`.
        - `contrast_scale`: The standard deviation of an image simulated
                            from `imaging_pipeline`, excluding the noise. By default,
                            `1.0`.
        """
        self.imaging_pipeline = imaging_pipeline
        self.variance = variance or Constant(1.0)
        self.contrast_scale = jnp.asarray(contrast_scale)

    @override
    def render(
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
        return self.contrast_scale * self.imaging_pipeline.render(
            normalize=True, get_real=get_real
        )

    @override
    def sample(
        self, key: PRNGKeyArray, *, get_real: bool = True
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
        N_pix = np.prod(pipeline.instrument_config.padded_shape)
        freqs = (
            pipeline.instrument_config.wrapped_padded_frequency_grid_in_angstroms.get()
        )
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(N_pix * self.variance(freqs))
        noise = pipeline.postprocess(
            std
            * jr.normal(key, shape=freqs.shape[0:-1]).at[0, 0].set(0.0).astype(complex),
            get_real=get_real,
        )
        image = self.render(get_real=get_real)
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
        N_pix = np.prod(pipeline.instrument_config.shape)
        freqs = pipeline.instrument_config.wrapped_frequency_grid_in_angstroms.get()
        # Compute the variance and scale up to be independent of the number of pixels
        variance = N_pix * self.variance(freqs)
        # Create simulated data
        simulated = self.render(get_real=False)
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_mode = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each fourier mode. Divide by the
        # number of pixels so that the likelihood is a sum over pixels in
        # real space (parseval's theorem)
        log_likelihood_per_mode = (
            squared_standard_normal_per_mode - jnp.log(2 * jnp.pi * variance) / 2
        ) / N_pix
        # Compute log-likelihood, throwing away the zero mode. Need to take care
        # to compute the loss function in fourier space for a real-valued function.
        log_likelihood = -1.0 * (
            jnp.sum(log_likelihood_per_mode[1:, 0])
            + 2 * jnp.sum(log_likelihood_per_mode[:, 1:])
        )

        return log_likelihood

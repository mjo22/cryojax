"""
Image formation models simulated from gaussian distributions.
"""

from typing import Optional, Any
from typing_extensions import override
from equinox import field

import numpy as np
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ._distribution import AbstractDistribution
from ...image.operators import FourierOperatorLike, Constant
from ...simulator import AbstractPipeline
from ...typing import Real_, RealImage, ComplexImage


class IndependentFourierGaussian(AbstractDistribution, strict=True):
    r"""
    A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.

    Attributes
    ----------
    variance :
        The gaussian variance function.
    """

    pipeline: AbstractPipeline
    variance: FourierOperatorLike
    contrast_scale: Real_ = field(converter=jnp.asarray)

    def __init__(
        self,
        pipeline: AbstractPipeline,
        variance: Optional[FourierOperatorLike] = None,
        contrast_scale: Optional[Real_] = None,
    ):
        self.pipeline = pipeline
        self.variance = variance or Constant(1.0)
        self.contrast_scale = contrast_scale or jnp.asarray(1.0)

    @override
    def sample(self, key: PRNGKeyArray, **kwargs: Any) -> RealImage:
        """Sample from the Gaussian noise model."""
        N_pix = np.prod(self.pipeline.scattering.config.padded_shape)
        freqs = self.pipeline.scattering.config.padded_frequency_grid_in_angstroms.get()
        # Compute the zero mean variance and scale up to be independent of the number of pixels
        std = jnp.sqrt(N_pix * self.variance(freqs))
        noise = std * jr.normal(key, shape=freqs.shape[0:-1]).at[0, 0].set(0.0)
        image = self.contrast_scale * self.pipeline.render(
            view_cropped=False, get_real=False
        )
        return self.pipeline.crop_and_apply_operators(image + noise, **kwargs)

    @override
    def log_probability(self, observed: ComplexImage) -> Real_:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        `observed` : The observed data in fourier space. `observed.shape`
                     must match `ImageConfig.padded_shape`.
        """
        pipeline = self.pipeline
        N_pix = np.prod(self.pipeline.scattering.config.padded_shape)
        padded_freqs = (
            pipeline.scattering.config.padded_frequency_grid_in_angstroms.get()
        )
        freqs = pipeline.scattering.config.frequency_grid_in_angstroms.get()
        if observed.shape != padded_freqs.shape[:-1]:
            raise ValueError("Shape of observed must match ImageConfig.padded_shape")
        # Compute the variance and scale up to be independent of the number of pixels
        variance = N_pix * self.variance(freqs)
        # Get residuals
        simulated = self.contrast_scale * pipeline.render(
            view_cropped=False, get_real=False
        )
        residuals = simulated - observed
        # Apply filters, crop, and mask
        residuals = pipeline.crop_and_apply_operators(residuals, get_real=False)
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

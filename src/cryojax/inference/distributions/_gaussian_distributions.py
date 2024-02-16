"""
Image formation models simulated from gaussian distributions.
"""

from typing import Optional, Any
from typing_extensions import override

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

    def __init__(
        self,
        pipeline: AbstractPipeline,
        variance: Optional[FourierOperatorLike] = None,
    ):
        self.pipeline = pipeline
        self.variance = variance or Constant(1.0)

    @override
    def sample(self, key: PRNGKeyArray, **kwargs: Any) -> RealImage:
        """Sample from the Gaussian noise model."""
        N_pix = np.prod(self.pipeline.scattering.config.padded_shape)
        freqs = self.pipeline.scattering.config.padded_frequency_grid_in_angstroms.get()
        # Compute the variance and scale up to be independent of the number of pixels
        variance = jnp.sqrt(N_pix) * self.variance(freqs)
        noise = variance * jr.normal(key, shape=freqs.shape[0:-1])
        image = self.pipeline.render(view_cropped=False, get_real=False)
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
        # Get residuals
        residuals = pipeline.render(view_cropped=False, get_real=False) - observed
        # Apply filters, crop, and mask
        residuals = pipeline.crop_and_apply_operators(residuals, get_real=False)
        # Compute the variance and scale up to be independent of the number of pixels
        variance = jnp.sqrt(N_pix) * self.variance(freqs)
        # Compute loss
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        # Divide by number of modes (parseval's theorem)
        loss = loss.real / residuals.size

        return loss

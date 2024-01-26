"""
Image formation models, equipped with probabilistic models.
"""

from __future__ import annotations

__all__ = ["AbstractDistribution", "IndependentFourierGaussian"]

from abc import abstractmethod
from typing import Optional, Any
from typing_extensions import override

import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import Module

from ..image.operators import FourierOperatorLike, Constant
from ..simulator.ice import GaussianIce
from ..simulator.detector import GaussianDetector
from ..simulator.pipeline import ImagePipeline
from ..typing import Real_, RealImage, ComplexImage, Image


class AbstractDistribution(Module):
    """
    An imaging pipeline equipped with a probabilistic model.
    """

    pipeline: ImagePipeline

    @abstractmethod
    def log_probability(self, observed: Image) -> Real_:
        """
        Evaluate the log-probability.

        Parameters
        ----------
        observed :
            The observed data in real or fourier space.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, key: PRNGKeyArray, **kwargs: Any) -> RealImage:
        """
        Sample from the distribution.

        Parameters
        ----------
        key :
            The RNG key or key(s). See ``ImagePipeline.sample`` for
            documentation.
        """
        raise NotImplementedError


class IndependentFourierGaussian(AbstractDistribution):
    r"""
    A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.

    If no variance model is explicitly passed, the variance is computed as

    .. math::
        Var[D(q)] + CTF(q)^2 Var[I(q)]

    where :math:`D(q)` and :math:`I(q)` are independent gaussian random variables in fourier
    space for the detector and ice, respectively, for a given fourier mode :math:`q`.

    Attributes
    ----------
    variance :
        The gaussian variance function. If not given, use the detector and ice noise
        models as described above.
    """

    variance: FourierOperatorLike

    def __init__(
        self,
        pipeline: ImagePipeline,
        variance: Optional[FourierOperatorLike] = None,
    ):
        self.pipeline = pipeline
        if variance is None:
            # Variance from detector
            if isinstance(pipeline.instrument.detector, GaussianDetector):
                variance = pipeline.instrument.detector.variance
            else:
                variance = Constant(0.0)
            # Variance from ice
            if isinstance(pipeline.solvent, GaussianIce):
                ctf = pipeline.instrument.optics.ctf
                variance += ctf * ctf * pipeline.solvent.variance
            if eqx.tree_equal(variance, Constant(0.0)):
                raise AttributeError(
                    "If variance is not given, the ImagePipeline must have either a GaussianDetector or GaussianIce model."
                )
        self.variance = variance

    @override
    def sample(self, key: PRNGKeyArray, **kwargs: Any) -> RealImage:
        """Sample from the Gaussian noise model."""
        freqs = (
            self.pipeline.scattering.manager.padded_frequency_grid_in_angstroms.get()
        )
        noise = self.variance(freqs) * jr.normal(key, shape=freqs.shape[0:-1])
        image = self.pipeline.render(view_cropped=False, get_real=False)
        return self.pipeline.crop_and_apply_operators(image + noise, **kwargs)

    @override
    def log_probability(self, observed: ComplexImage) -> Real_:
        """
        Evaluate the log-likelihood of the gaussian noise model.

        This evaluates the log probability in the super sampled
        coordinate system. Therefore, ``observed.shape`` must
        match ``ImageManager.padded_shape``.

        Parameters
        ----------
        observed :
           The observed data in fourier space. This must match
           the ImageManager.padded_shape shape.
        """
        pipeline = self.pipeline
        padded_freqs = (
            pipeline.scattering.manager.padded_frequency_grid_in_angstroms.get()
        )
        freqs = pipeline.scattering.manager.frequency_grid_in_angstroms.get()
        if observed.shape != padded_freqs.shape[:-1]:
            raise ValueError(
                "Shape of observed must match ImageManager.padded_shape"
            )
        # Get residuals
        residuals = (
            pipeline.render(view_cropped=False, get_real=False) - observed
        )
        # Apply filters, crop, and mask
        residuals = pipeline.crop_and_apply_operators(
            residuals, get_real=False
        )
        # Compute loss
        loss = jnp.sum(
            (residuals * jnp.conjugate(residuals)) / (2 * self.variance(freqs))
        )
        # Divide by number of modes (parseval's theorem)
        loss = loss.real / residuals.size

        return loss

"""
Image formation models, equipped with probabilistic models.
"""

from __future__ import annotations

__all__ = ["Distribution", "IndependentFourierGaussian"]

from abc import abstractmethod
from typing import Union, Optional
from typing_extensions import override
from functools import cached_property

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .noise import GaussianNoise
from .ice import NullIce, GaussianIce
from .detector import NullDetector, GaussianDetector
from .image import ImagePipeline, _PRNGKeyArrayLike
from ..utils import ifftn, fftn
from ..typing import Real_, RealImage
from ..core import Module, field


class Distribution(Module):
    """
    An imaging pipeline equipped with a probabilistic model.
    """

    pipeline: ImagePipeline = field()

    @abstractmethod
    def log_probability(self, observed: RealImage) -> Real_:
        """
        Evaluate the log-probability.

        Parameters
        ----------
        observed :
            The observed data in real space. This must be the same
            shape as ``scattering.shape``. Note that the user
            should preprocess the observed data before passing it
            to the image, such as applying the ``filters`` and
            ``masks``.
        """
        raise NotImplementedError

    def sample(self, key: Union[PRNGKeyArray, _PRNGKeyArrayLike]) -> RealImage:
        """
        Sample from the distribution.

        Parameters
        ----------
        key :
            The RNG key or key(s). See ``ImagePipeline.sample`` for
            documentation.
        """
        return self.pipeline.sample(key)


class IndependentFourierGaussian(Distribution):
    r"""
    A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.

    If no `GaussianNoise` model is explicitly passed, the variance is computed as

    .. math::
        Var[D(q)] + CTF(q)^2 Var[I(q)]

    where :math:`D(q)` and :math:`I(q)` are independent gaussian random variables in fourier
    space for the detector and ice, respectively, for a given fourier mode :math:`q`.

    Attributes
    ----------
    noise :
        The gaussian noise model. If not given, use the detector and ice noise
        models.
    """

    noise: Optional[GaussianNoise] = field(default=None)

    def __check_init__(self):
        if (
            not isinstance(self.pipeline.solvent, GaussianIce)
            and not isinstance(
                self.pipeline.instrument.detector, GaussianDetector
            )
        ) and self.noise is None:
            raise ValueError(
                "Either a GaussianIce, GaussianDetector, or GaussianNoise model are required."
            )

    @override
    def sample(self, key: Union[PRNGKeyArray, _PRNGKeyArrayLike]) -> RealImage:
        """Sample from the Gaussian noise model."""
        if self.noise is None:
            return super().sample(key)
        else:
            freqs = self.pipeline.scattering.padded_frequency_grid_in_angstroms
            noise = self.noise.sample(key, freqs)
            image = self.pipeline.render(view=False, get_real=False)
            return self.pipeline._filter_crop_mask(
                image + noise, is_real=False
            )

    @override
    def log_probability(self, observed: RealImage) -> Real_:
        """Evaluate the log-likelihood of the gaussian noise model."""
        # Get residuals
        residuals = fftn(self.pipeline.render() - observed)
        # Compute loss
        loss = jnp.sum(
            (residuals * jnp.conjugate(residuals)) / (2 * self.variance)
        )
        # Divide by number of modes (parseval's theorem)
        loss = loss.real / residuals.size

        return loss

    @cached_property
    def variance(self) -> Union[Real_, RealImage]:
        pipeline = self.pipeline
        # Gather frequency coordinates
        freqs = pipeline.scattering.frequency_grid_in_angstroms
        if self.noise is None:
            # Variance from detector
            if not isinstance(pipeline.instrument.detector, NullDetector):
                variance = pipeline.instrument.detector.variance(freqs)
            else:
                variance = 0.0
            # Variance from ice
            if not isinstance(pipeline.solvent, NullIce):
                ctf = pipeline.instrument.optics(
                    freqs, pose=pipeline.ensemble.pose
                )
                variance += ctf**2 * pipeline.solvent.variance(freqs)
            return variance
        else:
            return self.noise.variance(freqs)

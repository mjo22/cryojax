"""
Image formation models, equipped with probabilistic models.
"""

from __future__ import annotations

__all__ = ["Distribution", "GaussianImage"]

from abc import abstractmethod
from typing import Union, Optional
from typing_extensions import override
from functools import cached_property

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .ice import NullIce, GaussianIce
from .detector import NullDetector, GaussianDetector
from .image import ImagePipeline, _PRNGKeyArrayLike
from ..utils import fftn
from ..typing import Real_, Image, RealImage


class Distribution(ImagePipeline):
    """
    An imaging pipeline equipped with a probabilistic model.
    """

    @abstractmethod
    def log_probability(self, observed: RealImage) -> Real_:
        """
        Evaluate the log-probability.

        Attributes
        ----------
        observed :
            The observed data in real space. This must be the same
            shape as ``scattering.shape``. Note that the user
            should preprocess the observed data before passing it
            to the image, such as applying the ``filters`` and
            ``masks``.
        """
        raise NotImplementedError

    @override
    def __call__(
        self,
        *,
        key: Optional[Union[PRNGKeyArray, _PRNGKeyArrayLike]] = None,
        observed: Optional[RealImage] = None,
        view: bool = True,
    ) -> Union[Image, Real_]:
        """
        If ``observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        if key is not None:
            return self.sample(key, view=view)
        elif observed is not None:
            return self.log_probability(observed)
        else:
            return self.render(view=view)


class GaussianImage(Distribution):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.
    """

    def __check_init__(self):
        if not isinstance(self.solvent, GaussianIce) and not isinstance(
            self.instrument.detector, GaussianDetector
        ):
            raise ValueError(
                "Either GaussianIce or GaussianDetector are required."
            )

    @override
    def log_probability(self, observed: RealImage) -> Real_:
        """Evaluate the log-likelihood of the data given a parameter set."""
        # Get variance
        variance = self.variance
        # Get residuals
        residuals = fftn(self.render() - observed)
        # Crop redundant frequencies
        _, N2 = self.manager.shape
        z = N2 // 2 + 1
        residuals = residuals[:, :z]
        if not isinstance(variance, Real_):
            variance = variance[:, :z]
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        loss = loss.real / residuals.size

        return loss

    @cached_property
    def variance(self) -> Union[Real_, RealImage]:
        # Gather image configuration
        freqs = self.manager.freqs / self.pixel_size
        # Variance from detector
        if not isinstance(self.instrument.detector, NullDetector):
            variance = self.instrument.detector.variance(freqs)
        else:
            variance = 0.0
        # Variance from ice
        if not isinstance(self.solvent, NullIce):
            ctf = self.instrument.optics(freqs, pose=self.specimen.pose)
            variance += ctf**2 * self.solvent.variance(freqs)
        return variance

"""
Image formation models, equipped with log-likelihood functions.
"""

from __future__ import annotations

__all__ = ["GaussianImage"]

from typing import Union
from functools import cached_property

import jax.numpy as jnp

from .ice import NullIce, GaussianIce
from .detector import NullDetector, GaussianDetector
from .image import DetectorImage
from ..utils import fftn
from ..types import Real_, RealImage


class GaussianImage(DetectorImage):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.
    """

    def __check_init__(self):
        if not isinstance(self.instrument.ice, (NullIce, GaussianIce)):
            raise ValueError("A GaussianIce model is required.")
        if not isinstance(
            self.instrument.detector, (NullDetector, GaussianDetector)
        ):
            raise ValueError("A GaussianDetector model is required.")

    def log_probability(self) -> Real_:
        """Evaluate the log-likelihood of the data given a parameter set."""
        # Get variance
        variance = self.variance
        # Get residuals
        residuals = fftn(self.residuals)
        # Crop redundant frequencies
        _, N2 = self.scattering.shape
        z = N2 // 2 + 1
        residuals = residuals[:, :z]
        if not isinstance(variance, Real_):
            variance = variance[:, :z]
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        loss = (loss.real / residuals.size) / residuals.size

        return loss

    @cached_property
    def variance(self) -> Union[Real_, RealImage]:
        # Gather image configuration
        if self.instrument.detector.pixel_size is None:
            pixel_size = self.specimen.resolution
        else:
            pixel_size = self.instrument.detector.pixel_size
        freqs = self.scattering.freqs / pixel_size
        # Variance from detector
        if not isinstance(self.instrument.detector, NullDetector):
            variance = self.instrument.detector.variance(freqs)
        else:
            variance = 0.0
        # Variance from ice
        if not isinstance(self.instrument.ice, NullIce):
            ctf = self.instrument.optics(freqs, pose=self.instrument.pose)
            variance += ctf**2 * self.instrument.ice.variance(freqs)
        return variance

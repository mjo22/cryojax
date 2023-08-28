"""
Image formation models, equipped with log-likelihood functions.
"""

from __future__ import annotations

__all__ = ["GaussianImage"]

from typing import Optional, Union

import jax.numpy as jnp

from .noise import GaussianNoise
from .state import PipelineState
from .ice import NullIce
from .image import DetectorImage
from ..utils import fft
from ..core import dataclass, Array
from . import Specimen


@dataclass
class GaussianImage(DetectorImage):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.
    """

    def __post_init__(self):
        if not isinstance(self.state.ice, GaussianNoise):
            raise ValueError("A GaussianNoise Ice model is required.")
        if not isinstance(self.state.detector, GaussianNoise):
            raise ValueError("A GaussianNoise Detector model is required.")

    def variance(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
    ) -> Array:
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Gather image configuration
        freqs, resolution = scattering.freqs, specimen.resolution
        if hasattr(state.detector, "pixel_size"):
            pixel_size = state.detector.pixel_size
        else:
            pixel_size = resolution
        # Variance from detector
        variance = state.detector.variance(freqs / pixel_size)
        # Variance from ice
        if not isinstance(state.ice, NullIce):
            ctf = state.optics(freqs / pixel_size)
            variance += ctf**2 * state.ice.variance(freqs / pixel_size)
        return variance

    def log_likelihood(
        self,
        state: Optional[PipelineState] = None,
        specimen: Optional[Specimen] = None,
    ) -> Union[float, Array]:
        """Evaluate the log-likelihood of the data given a parameter set."""
        state = state or self.state
        specimen = specimen or self.specimen
        scattering = self.scattering
        # Get variance
        variance = self.variance(state=state, specimen=specimen)
        # Get residuals
        residuals = fft(self.residuals(state=state, specimen=specimen))
        # Crop redundant frequencies
        _, N2 = scattering.shape
        z = N2 // 2 + 1
        residuals = residuals[:, :z]
        if isinstance(variance, Array):
            variance = variance[:, :z]
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        loss = (loss.real / residuals.size) / residuals.size

        return loss

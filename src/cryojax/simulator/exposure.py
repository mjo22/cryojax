"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure", "UniformExposure"]

from abc import abstractmethod
from typing import Any, Union

import jax.numpy as jnp

from ..core import field, Module
from ..typing import RealImage, ImageCoords, Real_


class Exposure(Module):
    """
    Controls parameters related to variation in
    the image intensity.

    For example, this might include
    the incoming electron dose and radiation damage.
    """

    @abstractmethod
    def scaling(
        self, freqs: ImageCoords, **kwargs: Any
    ) -> Union[RealImage, Real_]:
        """
        Evaluate the intensity scaling.

        Arguments
        ----------
        freqs : The fourier space cartesian coordinates.
        """
        pass

    @abstractmethod
    def offset(
        self, freqs: ImageCoords, **kwargs: Any
    ) -> Union[RealImage, Real_]:
        """
        Evaluate the intensity offset.

        Arguments
        ----------
        freqs : The fourier space cartesian coordinates.
        """
        pass


class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def scaling(self, freqs: ImageCoords, **kwargs: Any) -> Real_:
        return jnp.asarray(1.0)

    def offset(self, freqs: ImageCoords, **kwargs: Any) -> Real_:
        return jnp.asarray(0.0)


class UniformExposure(Exposure):
    """
    Rescale the signal intensity uniformly.

    Attributes
    ----------
    N : Intensity scaling.
    mu: Intensity offset.
    """

    N: Real_ = field(default=1.0)
    mu: Real_ = field(default=0.0)

    def scaling(self, freqs: ImageCoords, **kwargs: Any) -> Real_:
        return self.N

    def offset(self, freqs: ImageCoords, **kwargs: Any) -> RealImage:
        N1, N2 = freqs.shape[0:-1]
        return jnp.zeros((N1, N2)).at[0, 0].set(N1 * N2 * self.mu)

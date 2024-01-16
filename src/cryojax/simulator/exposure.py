"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure"]

from equinox import Module
from typing import Any

from .manager import ImageManager
from ..image.operators import FourierOperatorLike, Constant, ZeroMode
from ..core import field
from ..typing import ComplexImage, ImageCoords


class Exposure(Module):
    """
    Controls parameters related to variation in
    the image intensity. This is implemented as a fourier
    space scaling and offset.

    NOTE: In the future this might include a model for radiation damage.
    """

    scaling: FourierOperatorLike = field(default_factory=Constant)
    offset: FourierOperatorLike = field(default_factory=ZeroMode)

    def __call__(
        self, image: ComplexImage, manager: ImageManager, **kwargs: Any
    ):
        """Evaluate the electron exposure model."""
        frequency_grid = manager.padded_frequency_grid_in_angstroms.get()
        return self.scaling(frequency_grid, **kwargs) * image + self.offset(
            frequency_grid, shape_in_real_space=manager.padded_shape, **kwargs
        )


class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def __init__(self):
        self.scaling = Constant(1.0)
        self.offset = Constant(0.0)

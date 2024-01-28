"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure"]

from equinox import Module, field

from .manager import ImageManager
from ..image import rfftn, irfftn
from ..image.operators import (
    RealOperatorLike,
    FourierOperatorLike,
    Constant,
)
from ..typing import ComplexImage


class Exposure(Module):
    """
    Controls parameters related to variation in
    the image intensity. This is implemented as a combination
    of real and fourier-space image operations.

    Attributes
    ----------
    dose :
        The dose delivered to the sample.
        This is modeled as a real-space function applied to the image.
    radiation :
        The amplitude attenuation function due to radiation damage.
        This is modeled as a fourier-space function applied to the image.
    """

    dose: RealOperatorLike = field(default_factory=Constant)
    radiation: FourierOperatorLike = field(default_factory=Constant)

    def __call__(
        self,
        image_at_exit_plane: ComplexImage,
        manager: ImageManager,
    ) -> ComplexImage:
        """Evaluate the electron exposure model."""
        coordinate_grid = manager.padded_coordinate_grid_in_angstroms.get()
        frequency_grid = manager.padded_frequency_grid_in_angstroms.get()
        if isinstance(self.dose, Constant):
            image_at_exit_plane *= self.dose(coordinate_grid)
        else:
            image_at_exit_plane = rfftn(
                self.dose(coordinate_grid)
                * irfftn(image_at_exit_plane, s=manager.padded_shape)
            )
        return self.radiation(frequency_grid) * image_at_exit_plane


class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def __init__(self):
        self.dose = Constant(1.0)
        self.radiation = Constant(1.0)

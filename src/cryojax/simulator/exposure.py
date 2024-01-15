"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure"]


from equinox import Module

from ..image.operators import FourierOperatorLike, Constant, ZeroMode
from ..core import field


class Exposure(Module):
    """
    Controls parameters related to variation in
    the image intensity.

    For example, this might include
    the incoming electron dose and radiation damage.
    """

    scaling: FourierOperatorLike = field(default_factory=Constant)
    offset: FourierOperatorLike = field(default_factory=ZeroMode)


class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def __init__(self):
        self.scaling = Constant(1.0)
        self.offset = Constant(0.0)

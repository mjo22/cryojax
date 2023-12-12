"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["Instrument"]

from ..core import Module, field

from .optics import Optics, NullOptics
from .exposure import Exposure, NullExposure
from .detector import Detector, NullDetector


class Instrument(Module):
    """
    An abstraction of an electron microscope.

    Attributes
    ----------
    optics :
        The model for the contrast transfer function.
    exposure :
        The model for the exposure to the electron beam.
    detector :
        The model of the detector.
    """

    optics: Optics = field(default_factory=NullOptics)
    exposure: Exposure = field(default_factory=NullExposure)
    detector: Detector = field(default_factory=NullDetector)

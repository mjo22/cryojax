"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["PipelineState"]

from ..core import Module, field

from .pose import Pose, EulerPose
from .ice import Ice, NullIce
from .optics import Optics, NullOptics
from .exposure import Exposure, NullExposure
from .detector import Detector, NullDetector


class PipelineState(Module):
    """
    A container for the state of the imaging pipeline.

    Attributes
    ----------
    pose :
        The pose of the specimen.
    ice :
        The model of the ice.
    optics :
        The CTF model.
    exposure :
        The model for intensity scaling.
    detector :
        The model of the detector.
    """

    pose: Pose = field(default_factory=EulerPose)
    ice: Ice = field(default_factory=NullIce)
    optics: Optics = field(default_factory=NullOptics)
    exposure: Exposure = field(default_factory=NullExposure)
    detector: Detector = field(default_factory=NullDetector)

"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["PipelineState"]

from ..core import dataclass, field, CryojaxObject
from .pose import Pose, EulerPose
from .ice import Ice, NullIce
from .optics import Optics, NullOptics
from .exposure import Exposure, NullExposure
from .detector import Detector, NullDetector


@dataclass
class PipelineState(CryojaxObject):
    """
    A container for the state of the imaging pipeline.

    Attributes
    ----------
    pose : `cryojax.simulator.Pose`
        The pose of the specimen.
    ice : `cryojax.simulator.Ice`
        The model of the ice.
    optics : `cryojax.simulator.OpticsModel`
        The CTF model.
    exposure : `cryojax.simulator.Exposure`
        The model for intensity scaling.
    detector : ``cryojax.simulator.Detector``
        The model of the detector.
    """

    pose: Pose = field(default=EulerPose())
    ice: Ice = field(default=NullIce())
    optics: Optics = field(default=NullOptics())
    exposure: Exposure = field(default=NullExposure())
    detector: Detector = field(default=NullDetector())

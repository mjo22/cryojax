"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["PipelineState"]

from jax import random

from ..core import dataclass, field, CryojaxObject
from .pose import EulerPose
from .ice import NullIce
from .optics import NullOptics
from .exposure import NullExposure
from .detector import NullDetector
from . import Pose, Ice, Optics, Exposure, Detector


@dataclass
class PipelineState(CryojaxObject):
    """
    PyTree container for the state of an ``ImageModel``.

    Attributes
    ----------
    pose : `cryojax.simulator.Pose`
        The image pose.
    ice : `cryojax.simulator.Ice`
        The model of the ice.
    optics : `cryojax.simulator.OpticsModel`
        The CTF model.
    exposure : `cryojax.simulator.Exposure`
        The model for intensity scaling.
    detector : ``cryojax.simulator.Detector``
        The model of the detector.
    """

    pose: Pose = field(default=EulerPose(), encode=Pose)
    ice: Ice = field(
        default=NullIce(key=random.PRNGKey(seed=1234)), encode=Ice
    )
    optics: Optics = field(default=NullOptics(), encode=Optics)
    exposure: Exposure = field(default=NullExposure(), encode=Exposure)
    detector: Detector = field(
        default=NullDetector(key=random.PRNGKey(seed=5678)), encode=Detector
    )

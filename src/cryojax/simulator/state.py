"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["ParameterDict", "ParameterState"]

import dataclasses
from typing import TypedDict, Optional

from jax import random

from ..core import dataclass, field, Scalar, Serializable
from .pose import EulerPose
from .ice import NullIce
from .optics import NullOptics
from .exposure import UniformExposure
from .detector import NullDetector
from . import Pose, Ice, Optics, Exposure, Detector


class ParameterDict(TypedDict):
    """
    Dictionary to facilitate JAX functional
    transformations.
    """

    # Pose parameters
    offset_x: Optional[Scalar]
    offset_y: Optional[Scalar]

    view_phi: Optional[Scalar]
    view_theta: Optional[Scalar]
    view_psi: Optional[Scalar]

    view_qw: Optional[Scalar]
    view_qx: Optional[Scalar]
    view_qy: Optional[Scalar]
    view_qz: Optional[Scalar]

    # CTF parameters
    defocus_u: Optional[Scalar]
    defocus_v: Optional[Scalar]
    defocus_angle: Optional[Scalar]
    voltage: Optional[Scalar]
    spherical_aberration: Optional[Scalar]
    amplitude_contrast: Optional[Scalar]
    phase_shift: Optional[Scalar]
    b_factor: Optional[Scalar]

    # Ice parameters
    kappa: Optional[Scalar]
    xi: Optional[Scalar]

    # Detector parameters
    alpha: Optional[Scalar]

    # Image intensity
    N: Optional[Scalar]
    mu: Optional[Scalar]


@dataclass
class ParameterState(Serializable):
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
    exposure: Exposure = field(default=UniformExposure(), encode=Exposure)
    detector: Detector = field(
        default=NullDetector(key=random.PRNGKey(seed=5678)), encode=Detector
    )

    def update(self, params: ParameterDict) -> ParameterState:
        """Return a new ParameterState based on a ParameterDict."""
        fields = dataclasses.fields(self)
        update = {}
        for field in fields:
            update[field.name] = {
                k: v
                for k, v in params.items()
                if k in getattr(self, field.name).data
            }
        return self.replace(
            pose=self.pose.replace(**update["pose"]),
            ice=self.ice.replace(**update["ice"]),
            optics=self.optics.replace(**update["optics"]),
            exposure=self.exposure.replace(**update["exposure"]),
            detector=self.detector.replace(**update["detector"]),
        )

"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["ParameterDict", "ParameterState"]

from typing import TypedDict, Optional

from ..types import dataclass, Scalar
from .optics import OpticsModel, NullOptics
from .cloud import Pose


class ParameterDict(TypedDict):
    """
    Dictionary to facilitate JAX functional
    transformations.
    """

    # Pose parameters
    view_phi: Optional[Scalar]
    view_theta: Optional[Scalar]
    view_psi: Optional[Scalar]
    offset_x: Optional[Scalar]
    offset_y: Optional[Scalar]

    # CTF parameters
    defocus_u: Optional[Scalar]
    defocus_v: Optional[Scalar]
    defocus_angle: Optional[Scalar]
    voltage: Optional[Scalar]
    spherical_aberration: Optional[Scalar]
    amplitude_contrast_ratio: Optional[Scalar]
    phase_shift: Optional[Scalar]
    b_factor: Optional[Scalar]


@dataclass
class ParameterState:
    """
    PyTree container for the state of an ``ImageModel``.

    Attributes
    ----------
    pose : Pose
        The image pose.
    optics : OpticsModel
        The CTF model parameters.
    """

    pose: Pose = Pose()
    optics: OpticsModel = NullOptics()

    def update(self, params: ParameterDict) -> ParameterState:
        """Return a new ParameterState based on a ParameterDict."""
        pose_update = {
            k: v for k, v in params.items() if hasattr(self.pose, k)
        }
        optics_update = {
            k: v for k, v in params.items() if hasattr(self.optics, k)
        }
        return self.replace(
            pose=self.pose.replace(**pose_update),
            optics=self.optics.replace(**optics_update),
        )

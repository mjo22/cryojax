"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["ParameterDict", "ParameterState"]

from typing import TypedDict, Optional

from ..types import dataclass, Scalar
from .transform import Pose, EulerPose
from .optics import OpticsModel, NullOptics
from .noise import Noise, NullNoise


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
    amplitude_contrast_ratio: Optional[Scalar]
    phase_shift: Optional[Scalar]
    b_factor: Optional[Scalar]


@dataclass
class ParameterState:
    """
    PyTree container for the state of an ``ImageModel``.

    Attributes
    ----------
    pose : `jax_2dtm.simulator.Pose`
        The image pose.
    optics : `jax_2dtm.simulator.OpticsModel`
        The CTF model.
    """

    pose: Pose = EulerPose()
    optics: OpticsModel = NullOptics()
    noise: Noise = NullNoise()

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

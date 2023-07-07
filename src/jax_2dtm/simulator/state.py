"""
Containers for storing parameters.
"""

from __future__ import annotations

__all__ = ["ParameterDict", "ParameterState"]

from typing import TypedDict, Optional
from ..types import dataclass
from .cloud import Pose


class ParameterDict(TypedDict):
    """
    Dictionary to facilitate JAX functional
    transformations.
    """

    view_phi: Optional[float]
    view_theta: Optional[float]
    view_psi: Optional[float]
    offset_x: Optional[float]
    offset_y: Optional[float]


@dataclass
class ParameterState:
    """
    Stores the state of an ``ImageModel``.

    Attributes
    ----------
    pose : Pose
        The image pose.
    """

    pose: Pose

    def update(self, params: ParameterDict) -> ParameterState:
        """
        Return a new ParameterState based on a
        ParameterDict.
        """
        pose_update = {
            k: v for k, v in params.items() if hasattr(self.pose, k)
        }
        return self.replace(pose=self.pose.replace(**pose_update))

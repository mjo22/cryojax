"""
Containers for storing parameter PyTrees.
"""

__all__ = ["ParameterDict", "ParameterState"]


from typing import TypedDict
from ..types import dataclass
from .cloud import Pose


class ParameterDict(TypedDict):
    """
    Dictionary to facilitate JAX functional
    transformations.
    """

    pose: dict


@dataclass
class ParameterState:
    """
    Stores the state of a ``simulator.ImageModel``
    """

    pose: Pose

    def update(self, params: ParameterDict):
        """
        Return a new ParameterState based on a
        ParameterDict.
        """
        return self.replace(pose=self.pose.replace(**params["pose"]))

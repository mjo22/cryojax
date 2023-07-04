"""
Containers for storing parameter PyTrees.
"""

__all__ = ["ParameterState"]


from ..types import dataclass
from .cloud import Pose


@dataclass
class ParameterState:
    """ """

    pose: Pose

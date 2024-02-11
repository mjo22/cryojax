"""
Base scattering potential representation.
"""

from abc import abstractmethod
from typing import Any
from typing_extensions import Self
from jaxtyping import PyTree
from equinox import Module

from .._pose import AbstractPose
from ...coordinates import get_not_coordinate_filter_spec


def is_potential_leaves_without_coordinates(element: Any) -> bool | PyTree[bool]:
    """Returns a filter spec that is ``True`` at the ``AbstractScatteringPotential``
    leaves, besides its coordinates.
    """
    if isinstance(element, AbstractScatteringPotential):
        return get_not_coordinate_filter_spec(element)
    else:
        return False


class AbstractScatteringPotential(Module, strict=True):
    """Abstract interface for an electron scattering potential."""

    @abstractmethod
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new `AbstractScatteringPotential` at the given pose.

        **Arguments:**

        - `pose`: The pose at which to view the `AbstractScatteringPotential`.
        """
        raise NotImplementedError

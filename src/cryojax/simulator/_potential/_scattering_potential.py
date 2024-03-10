"""
Base scattering potential representation.
"""

from abc import abstractmethod
from typing_extensions import Self
from equinox import Module

from .._pose import AbstractPose


class AbstractScatteringPotential(Module, strict=True):
    """Abstract interface for an electron scattering potential."""

    @abstractmethod
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new `AbstractScatteringPotential` at the given pose.

        **Arguments:**

        - `pose`: The pose at which to view the `AbstractScatteringPotential`.
        """
        raise NotImplementedError

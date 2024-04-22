"""
Base scattering potential representation.
"""

from abc import abstractmethod
from typing_extensions import Self

from equinox import Module

from .._pose import AbstractPose


class AbstractSpecimenPotential(Module, strict=True):
    """Abstract interface for the potential energy distribution of a
    biological specimen.
    """

    @abstractmethod
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new `AbstractScatteringPotential` at the given pose.

        **Arguments:**

        - `pose`: The pose at which to view the `AbstractSpecimenPotential`.
        """
        raise NotImplementedError

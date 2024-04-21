"""
Abstractions of biological specimen.
"""

from abc import abstractmethod
from functools import cached_property
from typing import Any, Optional
from typing_extensions import override

from equinox import AbstractVar, Module

from .._pose import AbstractPose, EulerAnglePose
from .._potential import AbstractScatteringPotential


class AbstractSpecimen(Module, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `potential`: The scattering potential of the specimen.
    - `pose`: The pose of the specimen.
    """

    potential: AbstractVar[Any]
    pose: AbstractVar[AbstractPose]

    @cached_property
    @abstractmethod
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the center of mass
        frame."""
        raise NotImplementedError

    @cached_property
    def potential_in_lab_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the lab frame."""
        return self.potential_in_com_frame.rotate_to_pose(self.pose)


class Specimen(AbstractSpecimen, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `potential`: The scattering potential representation of the
                    specimen as a single scattering potential object.
    """

    potential: AbstractScatteringPotential
    pose: AbstractPose

    def __init__(
        self,
        potential: AbstractScatteringPotential,
        pose: Optional[AbstractPose] = None,
    ):
        self.potential = potential
        self.pose = pose or EulerAnglePose()

    @cached_property
    @override
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the center of mass
        frame."""
        return self.potential

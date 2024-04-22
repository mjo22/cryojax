"""
Abstractions of ensembles of biological specimen.
"""

from abc import abstractmethod
from typing import Any, Optional
from typing_extensions import override

from equinox import AbstractVar, Module

from .._pose import AbstractPose, EulerAnglePose
from .._potential import AbstractSpecimenPotential
from .conformation import AbstractConformation


class AbstractPotentialEnsemble(Module, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `state_space`: The state space of the scattering potential.
    - `pose`: The pose of the scattering potential
    """

    state_space: AbstractVar[Any]
    pose: AbstractVar[AbstractPose]
    conformation: AbstractVar[Optional[AbstractConformation]]

    @abstractmethod
    def get_potential(self) -> AbstractSpecimenPotential:
        """Get the scattering potential in the center of mass
        frame."""
        raise NotImplementedError

    def get_potential_in_lab_frame(self) -> AbstractSpecimenPotential:
        """Get the scattering potential in the lab frame."""
        potential = self.get_potential_in_lab_frame()
        return potential.rotate_to_pose(self.pose)


class BaseEnsemble(AbstractPotentialEnsemble, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `potential`: The scattering potential representation of the
                    specimen as a single scattering potential object.
    """

    state_space: AbstractSpecimenPotential
    pose: AbstractPose
    conformation: None

    def __init__(
        self,
        state_space: AbstractSpecimenPotential,
        pose: Optional[AbstractPose] = None,
    ):
        self.state_space = state_space
        self.pose = pose or EulerAnglePose()
        self.conformation = None

    @override
    def get_potential(self) -> AbstractSpecimenPotential:
        """Get the scattering potential in the center of mass
        frame."""
        return self.state_space

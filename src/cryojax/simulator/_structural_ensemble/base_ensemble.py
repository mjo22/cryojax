"""
Abstractions of ensembles of biological specimen.
"""

from abc import abstractmethod
from typing import Any, Optional
from typing_extensions import override

from equinox import AbstractVar, Module

from .._pose import AbstractPose, EulerAnglePose
from .._potential_representation import AbstractPotentialRepresentation
from .base_conformation import AbstractConformationalVariable


class AbstractStructuralEnsemble(Module, strict=True):
    """A map from a pose and conformational variable to an `AbstractPotential`."""

    conformational_space: AbstractVar[Any]
    pose: AbstractVar[AbstractPose]
    conformation: AbstractVar[Optional[AbstractConformationalVariable]]

    @abstractmethod
    def get_potential_at_conformation(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential in the center of mass
        frame."""
        raise NotImplementedError

    def get_potential_in_lab_frame(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential in the lab frame."""
        potential = self.get_potential_at_conformation()
        return potential.rotate_to_pose(self.pose)


class SingleStructureEnsemble(AbstractStructuralEnsemble, strict=True):
    """An "ensemble" with one conformation."""

    conformational_space: AbstractPotentialRepresentation
    pose: AbstractPose
    conformation: None

    def __init__(
        self,
        conformational_space: AbstractPotentialRepresentation,
        pose: Optional[AbstractPose] = None,
    ):
        """**Arguments:**

        - `conformational_space`: The scattering potential representation of the
                         specimen as a single scattering potential object.
        - `pose`: The pose of the specimen.
        """
        self.conformational_space = conformational_space
        self.pose = pose or EulerAnglePose()
        self.conformation = None

    @override
    def get_potential_at_conformation(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential in the center of mass
        frame.
        """
        return self.conformational_space

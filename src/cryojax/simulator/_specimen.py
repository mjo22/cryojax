"""
Abstractions of biological specimen.
"""

from abc import abstractmethod
from typing import Optional, Any
from functools import cached_property
from typing_extensions import override
from equinox import AbstractVar

import jax
from equinox import Module

from ._potential import AbstractScatteringPotential
from ._pose import AbstractPose, EulerAnglePose
from ._conformation import AbstractConformation, DiscreteConformation


class AbstractSpecimen(Module, strict=True):
    """
    Abstraction of a of biological specimen.

    Attributes
    ----------
    potential :
        The scattering potential of the specimen.
    pose :
        The pose of the specimen.
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

    Attributes
    ----------
    potential :
        The scattering potential representation of the
        specimen as a single scattering potential object.
    pose :
        The pose of the specimen.
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


class AbstractEnsemble(AbstractSpecimen, strict=True):
    """
    Abstraction of an ensemble of a biological specimen which can
    occupy different conformations.

    Attributes
    ----------
    potential :
        A tuple of scattering potential representations.
    pose :
        The pose of the specimen.
    conformation :
        The conformation at which to evaluate the scattering potential.
    """

    conformation: AbstractVar[AbstractConformation]


class DiscreteEnsemble(AbstractEnsemble, strict=True):
    """
    Abstraction of an ensemble with discrete conformational
    heterogeneity.

    Attributes
    ----------
    potential :
        A tuple of scattering potential representations.
    pose :
        The pose of the specimen.
    conformation :
        A conformation with a discrete index at which to evaluate
        the scattering potential tuple.
    """

    potential: tuple[AbstractScatteringPotential, ...]
    pose: AbstractPose
    conformation: DiscreteConformation

    def __init__(
        self,
        potential: tuple[AbstractScatteringPotential, ...],
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformation] = None,
    ):
        self.potential = potential
        self.pose = pose or EulerAnglePose()
        self.conformation = conformation or DiscreteConformation(0)

    @cached_property
    @override
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential at configured conformation."""
        funcs = [lambda i=i: self.potential[i] for i in range(len(self.potential))]
        potential = jax.lax.switch(self.conformation.get(), funcs)

        return potential

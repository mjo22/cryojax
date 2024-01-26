"""
Abstractions of biological specimen.
"""

__all__ = [
    "SpecimenT",
    "EnsembleT",
    "AbstractSpecimen",
    "AbstractEnsemble",
    "Specimen",
    "DiscreteEnsemble",
]

from abc import abstractmethod
from typing import Optional, TypeVar, Any
from functools import cached_property
from typing_extensions import override
from equinox import AbstractVar

import jax
from equinox import Module

from .density import AbstractElectronDensity
from .pose import AbstractPose, EulerPose
from .conformation import AbstractConformation, DiscreteConformation


SpecimenT = TypeVar("SpecimenT", bound="AbstractSpecimen")
EnsembleT = TypeVar("EnsembleT", bound="AbstractEnsemble")


class AbstractSpecimen(Module):
    """
    Abstraction of a of biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen.
    pose :
        The pose of the specimen.
    """

    density: AbstractVar[Any]
    pose: AbstractPose

    @cached_property
    @abstractmethod
    def density_in_com_frame(self) -> AbstractElectronDensity:
        """Get the electron density in the center of mass
        frame."""
        raise NotImplementedError

    @cached_property
    def density_in_lab_frame(self) -> AbstractElectronDensity:
        """Get the electron density in the lab frame."""
        return self.density_in_com_frame.rotate_to_pose(self.pose)


class Specimen(AbstractSpecimen):
    """
    Abstraction of a of biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen as a single electron density object.
    pose :
        The pose of the specimen.
    """

    density: AbstractElectronDensity
    pose: AbstractPose

    def __init__(
        self,
        density: AbstractElectronDensity,
        pose: Optional[AbstractPose] = None,
    ):
        self.density = density
        self.pose = pose or EulerPose()

    @cached_property
    @override
    def density_in_com_frame(self) -> AbstractElectronDensity:
        """Get the electron density in the center of mass
        frame."""
        return self.density


class AbstractEnsemble(AbstractSpecimen):
    """
    Abstraction of an ensemble of a biological specimen which can
    occupy different conformations.

    Attributes
    ----------
    density :
        A tuple of electron density representations.
    pose :
        The pose of the specimen.
    conformation :
        The conformation at which to evaluate the ElectronDensity.
    """

    density: AbstractVar[Any]
    pose: AbstractPose
    conformation: AbstractVar[AbstractConformation]


class DiscreteEnsemble(AbstractEnsemble):
    """
    Abstraction of an ensemble with discrete conformational
    heterogeneity.

    Attributes
    ----------
    density :
        A tuple of electron density representations.
    pose :
        The pose of the specimen.
    conformation :
        A conformation with a discrete index at which to evaluate
        the electron density tuple.
    """

    density: tuple[AbstractElectronDensity, ...]
    pose: AbstractPose
    conformation: DiscreteConformation

    def __init__(
        self,
        density: tuple[AbstractElectronDensity, ...],
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformation] = None,
    ):
        self.density = density
        self.pose = pose or EulerPose()
        self.conformation = conformation or DiscreteConformation(0)

    @cached_property
    @override
    def density_in_com_frame(self) -> AbstractElectronDensity:
        """Get the electron density at configured conformation."""
        funcs = [lambda i=i: self.density[i] for i in range(len(self.density))]
        density = jax.lax.switch(self.conformation.get(), funcs)

        return density

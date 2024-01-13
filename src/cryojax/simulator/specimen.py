"""
Abstractions of a biological specimen.
"""

from __future__ import annotations

__all__ = ["SpecimenLike", "Specimen", "Ensemble", "Conformation"]

from abc import abstractmethod
from typing import Optional
from typing_extensions import override
from functools import cached_property

import jax
import jax.numpy as jnp
from equinox import Module

from .density import ElectronDensity
from .pose import Pose, EulerPose
from ..core import field
from ..typing import Int_


class SpecimenLike(Module):
    """
    Base class for things at act like biological specimen.
    """

    @abstractmethod
    def get_density(self) -> ElectronDensity:
        """Get the ElectronDensity at the configured state."""
        raise NotImplementedError


class Specimen(SpecimenLike):
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

    density: ElectronDensity
    pose: Pose

    def __init__(
        self,
        density: ElectronDensity,
        pose: Optional[Pose] = None,
    ):
        self.density = density
        self.pose = pose or EulerPose()

    def __check_init__(self):
        if self.density.n_indexed_dims != 0:
            raise AttributeError("ElectronDensity.n_indexed_dims must be 0.")

    @cached_property
    def density_at_pose(self) -> ElectronDensity:
        """Get the electron density at the configured pose."""
        return self.density.rotate_to_pose(self.pose)

    @override
    def get_density(self) -> ElectronDensity:
        """Get the ElectronDensity."""
        return self.density_at_pose


class Ensemble(SpecimenLike):
    """
    Abstraction of an ensemble of biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen.
    pose :
        The pose of the specimen.
    conformation :
        The conformation at which to evaluate the ElectronDensity.
    """

    density: ElectronDensity
    pose: Pose
    conformation: Conformation

    def __init__(
        self,
        density: ElectronDensity,
        pose: Optional[Pose] = None,
        conformation: Optional[Conformation] = None,
    ):
        self.density = density
        self.pose = pose or EulerPose()
        self.conformation = conformation or Conformation(0)

    def __check_init__(self):
        if self.density.n_indexed_dims != 1:
            raise AttributeError(
                "ElectronDensity.n_indexed_dims must be 1 to evaluate at a density at a conformation."
            )

    @cached_property
    def density_at_conformation_and_pose(self) -> ElectronDensity:
        """Get the electron density at the configured pose and conformation."""
        funcs = [lambda i=i: self.density[i] for i in range(len(self.density))]
        density = jax.lax.switch(self.conformation.get(), funcs)

        return density.rotate_to_pose(self.pose)

    @override
    def get_density(self) -> ElectronDensity:
        """Get the ElectronDensity."""
        return self.density_at_conformation_and_pose


class Conformation(Module):
    """
    A conformational variable wrapped in a Module.
    """

    _value: Int_ = field(converter=jnp.asarray)

    def get(self) -> Int_:
        return self._value

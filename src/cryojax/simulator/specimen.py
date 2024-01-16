"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = [
    "SpecimenT",
    "Specimen",
    "Ensemble",
    "Conformation",
]

from typing import Optional, TypeVar
from functools import cached_property
from typing_extensions import override

import jax
import jax.numpy as jnp
from equinox import Module

from .density import ElectronDensity
from .pose import Pose, EulerPose
from ..core import field
from ..typing import Int_


SpecimenT = TypeVar("SpecimenT", bound="Specimen")


class Specimen(Module):
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
        if density.n_indexed_dims != 0:
            raise AttributeError("ElectronDensity.n_indexed_dims must be 0.")
        self.density = density
        self.pose = pose or EulerPose()

    @cached_property
    def density_in_com_frame(self) -> ElectronDensity:
        """Get the electron density in the center of mass
        frame."""
        return self.density

    @cached_property
    def density_in_lab_frame(self) -> ElectronDensity:
        """Get the electron density in the lab frame."""
        return self.density_in_com_frame.rotate_to_pose(self.pose)


class Ensemble(Specimen):
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

    conformation: Conformation

    def __init__(
        self,
        density: ElectronDensity,
        pose: Optional[Pose] = None,
        conformation: Optional[Conformation] = None,
    ):
        if density.n_indexed_dims != 1:
            raise AttributeError(
                "ElectronDensity.n_indexed_dims must be 1 to evaluate at a density at a conformation."
            )
        self.density = density
        self.pose = pose or EulerPose()
        self.conformation = conformation or Conformation(0)

    @cached_property
    @override
    def density_in_com_frame(self) -> ElectronDensity:
        """Get the electron density at configured conformation."""
        funcs = [lambda i=i: self.density[i] for i in range(len(self.density))]
        density = jax.lax.switch(self.conformation.get(), funcs)

        return density


class Conformation(Module):
    """
    A conformational variable wrapped in a Module.
    """

    _value: Int_ = field(converter=jnp.asarray)

    def get(self) -> Int_:
        return self._value

"""
Abstractions of biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen", "Ensemble"]

from typing import Any

from .density import ElectronDensity
from .pose import Pose, EulerPose
from .conformation import Discrete
from ..core import field, Module
from ..typing import Real_


class Specimen(Module):
    """
    Abstraction of a biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen.
    resolution :
        Rasterization resolution. This is in
        dimensions of length.
    conformation :
        The conformational variable at which to evaulate
        the electron density. This does not do anything in
        the specimen base class and should be overwritten
        in subclasses.
    pose :
        The pose of the specimen.
    """

    density: ElectronDensity = field()
    resolution: Real_ = field()
    conformation: Any = field(default=None)

    pose: Pose = field(default_factory=EulerPose)

    @property
    def realization(self) -> ElectronDensity:
        """View the electron density at the pose."""
        return self.density.view(self.pose)


class Ensemble(Specimen):
    """
    A biological specimen at a discrete mixture of conformations.

    conformation :
        The discrete conformational variable at which to evaulate
        the electron density.
    """

    density: list[ElectronDensity] = field()
    conformation: Discrete = field(default_factory=Discrete)

    def __check_init__(self):
        coordinate = self.conformation.coordinate
        if not (-len(self.density) <= coordinate < len(self.density)):
            raise ValueError("The conformational coordinate is out-of-bounds.")

    @property
    def realization(self) -> ElectronDensity:
        """Sample the electron density at the configured conformation."""
        return self.density[self.conformation.coordinate].view(self.pose)

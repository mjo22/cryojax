"""
Abstractions of a biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen"]

from typing import Any
from functools import cached_property

from ..density import ElectronDensity
from ..pose import Pose, EulerPose
from ...core import field, Module
from ...typing import Real_


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
    def n_conformations(self) -> int:
        return 1

    @cached_property
    def realization(self) -> ElectronDensity:
        """View the electron density at the pose."""
        return self.density

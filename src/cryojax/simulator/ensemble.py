"""
Abstractions of a biological specimen.
"""

from __future__ import annotations

__all__ = ["Ensemble"]

from typing import Optional
from functools import cached_property

import jax

from .density import ElectronDensity
from .pose import Pose, EulerPose
from ..core import field, Module
from ..typing import Int_


class Ensemble(Module):
    """
    Abstraction of an ensemble of biological specimen.

    Attributes
    ----------
    density :
        The electron density representation of the
        specimen.
    conformation :
        The conformational variable at which to evaulate
        the electron density. Use this variable when, for example,
        the specimen ``ElectronDensity`` is constructed
        with ``ElectronDensity.from_stack``.
    pose :
        The pose of the specimen.
    """

    density: ElectronDensity
    pose: Pose = field(default_factory=EulerPose)
    conformation: Optional[Int_] = None

    def __check_init__(self):
        if self.density.n_stacked_dims != 0 and self.conformation is None:
            raise ValueError(
                "The conformation must be set if the ElectronDensity has a stacked dimension."
            )
        if self.density.n_stacked_dims != 1 and self.conformation is not None:
            raise ValueError(
                "If the conformation is set, the number of stacked dimensions of the ElectronDensity must be one."
            )

    @cached_property
    def realization(self) -> ElectronDensity:
        """Get the electron density at the configured pose and conformation."""
        if self.conformation is None:
            density = self.density
        else:
            funcs = [
                lambda i=i: self.density[i] for i in range(len(self.density))
            ]
            density = jax.lax.switch(self.conformation, funcs)
        return density.rotate_to_pose(self.pose)

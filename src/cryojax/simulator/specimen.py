"""
Abstractions of a biological specimen.
"""

from __future__ import annotations

__all__ = ["Specimen"]

from typing import Optional

import jax

from .density import ElectronDensity
from .pose import Pose, EulerPose
from ..core import field, Module
from ..typing import Real_, Int_


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
        the electron density. Use this variable when
        the specimen ``ElectronDensity`` is constructed
        with ``ElectronDensity.from_stack``.
    pose :
        The pose of the specimen.
    """

    density: ElectronDensity = field()
    resolution: Real_ = field()
    pose: Pose = field(default_factory=EulerPose)
    conformation: Optional[Int_] = field(default=None)

    def get_density(self) -> ElectronDensity:
        """Get the electron density at the configured pose and conformation."""
        if self.conformation is None:
            density = self.density
        else:
            funcs = [
                lambda i=i: self.density[i] for i in range(len(self.density))
            ]
            density = jax.lax.switch(self.conformation, funcs)
        return density.rotate_to(self.pose)

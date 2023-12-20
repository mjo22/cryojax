"""
A biological specimen with conformational heterogeneity.
"""

from __future__ import annotations

__all__ = ["Ensemble"]

from functools import cached_property

import jax

from ._specimen import Specimen
from ..density import ElectronDensity
from ..conformation import Discrete
from ...core import field


class Ensemble(Specimen):
    """
    A biological specimen at a discrete mixture of conformations.

    Attributes
    ----------
    density :
        A voxel-based electron density whose leading axis indexes different
        conformations.
    conformation :
        The discrete conformational variable at which to evaulate
        the electron density.
    """

    conformation: Discrete = field(default_factory=Discrete)

    def __check_init__(self):
        coordinate = self.conformation.coordinate
        if not self.density.is_stacked:
            raise ValueError("Ensemble requires a stacked electron density")
        if not (-self.n_conformations <= coordinate < self.n_conformations):
            raise ValueError("The conformational coordinate is out-of-bounds.")

    @property
    def n_conformations(self) -> int:
        return len(self.density)

    @cached_property
    def realization(self) -> ElectronDensity:
        """Sample the electron density at the configured conformation."""
        funcs = [
            lambda i=i: self.density[i] for i in range(self.n_conformations)
        ]
        density = jax.lax.switch(self.conformation.coordinate, funcs)
        return density.rotate_to(self.pose)

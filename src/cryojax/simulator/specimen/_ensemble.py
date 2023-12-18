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
    def n_conformations(self) -> int:
        return len(self.density)

    @cached_property
    def realization(self) -> ElectronDensity:
        """Sample the electron density at the configured conformation."""
        funcs = [
            lambda i=i: self.density[i] for i in range(self.n_conformations)
        ]
        density = jax.lax.switch(self.conformation.coordinate, funcs)
        return density.view(self.pose)

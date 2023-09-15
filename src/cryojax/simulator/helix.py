"""
Abstractions of helical filaments
"""

from __future__ import annotations

__all__ = ["Helix"]

from typing import Any, Annotated, Optional, Callable

import numpy as np

from .specimen import Specimen
from .conformation import Conformation
from .pose import Pose
from .scattering import ScatteringConfig

from ..core import Parameter, dataclass, field, Array, CryojaxObject

Lattice = Annotated[np.ndarray, (..., 3), np.floating]
Poses = Annotated[np.ndarray, (...), Pose]
Conformations = Annotated[np.ndarray, (...), Conformation]


@dataclass
class Helix(CryojaxObject):
    """
    Abstraction of a helical filament.

    Attributes
    ----------
    subunit : `cryojax.simulator.Specimen`
        The helical subunit.
    rise : `cryojax.core.Parameter`
        The helical rise.
    twist : `cryojax.core.Parameter`
        The helical twist.
    n_subunit : `int`
        The number of subunits in the lattice.
    """

    subunit: Specimen = field()
    rise: Parameter = field()
    twist: Parameter = field()

    n_subunit: int = field(pytree_node=False)
    conformations: Optional[Callable[[Lattice], Conformations]] = field(
        pytree_node=False, default=None
    )

    lattice: Lattice = field(pytree_node=False, init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "lattice",
            compute_lattice(self.rise, self.twist, self.n_subunit),
        )

    def scatter(
        self, scattering: ScatteringConfig, pose: Pose, **kwargs: Any
    ) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        The input and output of this method should identically
        match that of ``Specimen.scatter``.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration.
        pose : `cryojax.simulator.Pose`
            The imaging pose.
        """
        raise NotImplementedError

    @property
    def resolution(self) -> Parameter:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution


def compute_lattice(rise: float, twist: float, n_subunit: int) -> Lattice:
    """
    Compute the lattice points for a given
    helical rise and twist.

    rise : `float`
        The helical rise.
    twist : `float`
        The helical twist.
    n_subunit : `int`
        The number of subunits in the lattice.
    """
    return np.zeros((n_subunit, 3))

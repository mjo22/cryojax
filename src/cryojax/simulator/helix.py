"""
Abstractions of helical filaments
"""

from __future__ import annotations

__all__ = ["Helix"]

from typing import Any, Union, Annotated, Optional, Callable
from functools import partial

import jax.numpy as jnp

from .specimen import Specimen
from .pose import Pose
from .scattering import ScatteringConfig

from ..core import Parameter, dataclass, field, Array, CryojaxObject

Lattice = Annotated[Array, (..., 3), jnp.floating]
"""Type hint for array where each element is a lattice coordinate."""

Conformations = Annotated[Array, (...), Union[jnp.floating, int]]
"""Type hint for array where each element updates a Conformation."""


@partial(dataclass, kw_only=True)
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
    lattice : `Lattice`
        The 3D cartesian lattice coordinates for each subunit.
    conformations : `Conformations` or `Callable[[Lattice], Conformations]`, optional
        The conformation of `subunit` at each lattice sitte.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
    """

    subunit: Specimen = field()
    rise: Parameter = field()
    twist: Parameter = field()
    conformations: Optional[
        Union[Conformations, Callable[[Lattice], Conformations]]
    ] = field(default=None)

    n_subunit: int = field(pytree_node=False)
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

    @property
    def draw(self) -> Conformations:
        """Return an array where each elements updates a
        conformation object."""
        if isinstance(self.conformations, Callable):
            return self.conformations(self.lattice)
        else:
            return self.conformations


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
    return jnp.zeros((n_subunit, 3))

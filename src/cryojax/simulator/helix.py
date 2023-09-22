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
from .optics import Optics
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

    This class acts just like a ``Specimen``, however
    it assembles a helix from a subunit.

    Attributes
    ----------
    subunit : `cryojax.simulator.Specimen`
        The helical subunit.
    rise : `cryojax.core.Parameter` or `cryojax.core.Array`
        The helical rise. This has dimensions
        of length.
    twist : `cryojax.core.Parameter` or `cryojax.core.Array`
        The helical twist, given in degrees if
        ``degrees = True`` and radians otherwise.
    repeat : `cryojax.core.Array`, shape `(3,)` or `(n_repeat, 3)`
        The displacement vector between two subunits,
        longitudinally in contact.
    conformations : `Conformations` or `Callable[[Lattice], Conformations]`, optional
        The conformation of `subunit` at each lattice sitte.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
    n_repeat : `int`
        The number of longitudinal repeats of the helix.
        By default, ``1``.
    degrees : `bool`
        Whether or not the helical repeat is given in
        degrees. By default, ``True``.
    lattice : `Lattice`
        The 3D cartesian lattice coordinates for each subunit.
    """

    subunit: Specimen = field()
    rise: Parameter = field()
    twist: Parameter = field()
    repeat: Parameter = field()
    conformations: Optional[
        Union[Conformations, Callable[[Lattice], Conformations]]
    ] = field(default=None)

    n_repeat: int = field(pytree_node=False, default=1)
    degrees: bool = field(pytree_node=False, default=True)
    lattice: Lattice = field(pytree_node=False, init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "lattice",
            compute_lattice(self.rise, self.twist, self.repeat, self.n_repeat),
        )

    def scatter(
        self,
        scattering: ScatteringConfig,
        pose: Pose,
        optics: Optional[Optics] = None,
        **kwargs: Any,
    ) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        The input and output of this method should identically
        match that of ``Specimen.scatter``.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration for the subunit.
        pose : `cryojax.simulator.Pose`
            The center of mass imaging pose of the helix.
        optics : `cryojax.simulator.Optics`, optional
            The instrument optics.
        """
        image = self.subunit.scatter(scattering, pose, optics=optics, **kwargs)

        return image

    @property
    def resolution(self) -> Parameter:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution

    @property
    def draw(self) -> Conformations:
        """Return an array where each element updates a
        Conformation."""
        if isinstance(self.conformations, Callable):
            return self.conformations(self.lattice)
        else:
            return self.conformations


def compute_lattice(
    rise: float, twist: float, repeat: Array, n_repeat: int
) -> Lattice:
    """
    Compute the lattice points for a given
    helical rise and twist.

    rise : `float`
        The helical rise.
    twist : `float`
        The helical twist.
    repeat : `cryojax.core.Array`, shape `(3,)`
        The displacement vector between two subunits,
        longitudinally in contact.
    n_repeat : `int`
        The number of longitudinal repeats of the helix.
        By default, ``1``.
    """
    pitch = 2 * jnp.pi * rise / twist  # Helical pitch (distance between turns)
    d = jnp.linalg.norm(repeat)
    turns_per_repeat = int(d / pitch)  # Number of turns
    subunits_per_repeat = int(d / rise)  # Number of points per turn
    n_subunits = turns_per_repeat * subunits_per_repeat
    return jnp.zeros((n_subunits, 3))

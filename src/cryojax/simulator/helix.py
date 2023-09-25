"""
Abstractions of helical filaments
"""

from __future__ import annotations

__all__ = ["Helix", "compute_lattice"]

from typing import Any, Union, Annotated, Optional, Callable
from functools import partial

import jax
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
    rise : `Parameter` or `Array`, shape `(n_subunits,)`
        The helical rise. This has dimensions
        of length.
    twist : `Parameter` or `Array`, shape `(n_subunits,)`
        The helical twist, given in degrees if
        ``degrees = True`` and radians otherwise.
    radius : `Parameter` or `Array`, shape `(n_subunits,)`
        The radius of the helix.
    conformations : `Array` or `Callable[[Lattice], Array]`, optional
        The conformation of `subunit` at each lattice site.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
        In either case, the `Array` should be shape `(n_start*n_subunits,)`.
    n_start : `int`
        The start number of the helix. By default, ``1``.
    n_subunits : `int`, optional
        The number of subunits in the assembly.
    degrees : `bool`
        Whether or not the helical twist is given in
        degrees. By default, ``True``.
    lattice : `Lattice`
        The 3D cartesian lattice coordinates for each subunit.
    """

    subunit: Specimen = field()
    rise: Parameter = field()
    twist: Parameter = field()
    radius: Parameter = field(default=1)
    conformations: Optional[
        Union[Conformations, Callable[[Lattice], Conformations]]
    ] = field(default=None)

    n_start: int = field(pytree_node=False, default=1)
    n_subunits: Optional[int] = field(pytree_node=False, default=None)
    degrees: bool = field(pytree_node=False, default=True)
    lattice: Lattice = field(pytree_node=False, init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "lattice",
            compute_lattice(
                self.rise,
                self.twist,
                radius=self.radius,
                n_start=self.n_start,
                n_subunits=self.n_subunits,
                degrees=self.degrees,
            ),
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
    rise: float,
    twist: float,
    radius: float = 1.0,
    n_start: int = 1,
    n_subunits: Optional[int] = None,
    *,
    degrees: bool = True,
) -> Lattice:
    """
    Compute the lattice points of a helix for a given
    rise, twist, radius, and start number.

    Parameters
    ----------
    rise : `float` or `Array`, shape (n_subunits,)
        The helical rise.
    twist : `float` or `Array`, shape (n_subunits,)
        The helical twist.
    radius : `float` or `Array`, shape (n_subunits,)
        The radius of the helix.
    n_start : `int`
        The start number of the helix.
    n_subunits : `int`, optional
        The number of subunits in the assembly for
        a single helix. The total number of subunits
        is really equal to ``n_start * n_subunits``.
        By default, ``2 * jnp.pi / twist``.
    degrees : `bool`
        Whether or not the angular parameters
        are given in degrees or radians.

    Returns
    -------
    lattice : `Array`, shape (n_start*n_subunits, 3)
        The helical lattice.
    """
    # Convert to radians
    if degrees:
        twist = jnp.deg2rad(twist)
    # If the number of subunits is not given, compute for one helix
    if n_subunits is None:
        n_subunits = int(2 * jnp.pi / twist)
    # Rotational symmetry between helices due to the start number
    symmetry_angles = jnp.array(
        [2 * jnp.pi * n / n_start for n in range(n_start)]
    )

    def compute_helical_coordinates(symmetry_angle):
        """
        Get the coordinates for a given
        helix, where the x and y coordinates
        are rotated by an angle.
        """
        theta = jnp.arange(n_subunits, dtype=float) * twist
        x_0 = radius * jnp.cos(theta)
        y_0 = radius * jnp.sin(theta)
        c, s = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R = jnp.array(((c, s), (-s, c)), dtype=float)
        x, y = R @ jnp.stack((x_0, y_0))
        z = rise * jnp.arange(-n_subunits / 2, n_subunits / 2, 1, dtype=float)
        return jnp.stack((x, y, z), axis=-1)

    # The helical coordinates for all sub-helices
    lattice = jax.vmap(compute_helical_coordinates)(symmetry_angles)
    return lattice.reshape((n_start * n_subunits, 3))

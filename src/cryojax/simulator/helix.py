"""
Abstractions of helical filaments.
"""

from __future__ import annotations

__all__ = ["Helix", "compute_lattice_positions", "compute_lattice_rotations"]

from typing import Union, Optional, Callable
from jaxtyping import Array, Float, Int
from functools import cached_property

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from .specimen import Specimen
from .pose import Pose, EulerPose

from ..core import field, Module
from ..types import Real_, RealVector

Positions = Float[Array, "N 3"]
"""Type hint for array where each element is a lattice coordinate."""

Rotations = Float[Array, "N 3 3"]
"""Type hint for array where each element is a rotation matrix."""

Conformations = Union[Float[Array, "N"], Int[Array, "N"]]
"""Type hint for array where each element updates a Conformation."""


class Helix(Module):
    """
    Abstraction of a helical filament.

    This class acts just like a ``Specimen``, however
    it assembles a helix from a subunit.

    The screw axis is taken to be in the center of the
    image, pointing out-of-plane (i.e. along the z direction).

    Attributes
    ----------
    subunit :
        The helical subunit. It is important to set the the initial pose
        of the initial subunit here. This is in the center of mass frame
        of the helix with the screw axis pointing in the z-direction.
    rise :
        The helical rise. This has dimensions
        of length.
    twist :
        The helical twist, given in degrees if
        ``degrees = True`` and radians otherwise.
    pose :
        The center of mass pose of the helix.
    conformations :
        The conformation of `subunit` at each lattice site.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
        In either case, the `Array` should be shape `(n_start*n_subunits,)`.
    n_start :
        The start number of the helix. By default, ``1``.
    n_subunits :
        The number of subunits in the assembly.
    degrees :
        Whether or not the helical twist is given in
        degrees. By default, ``True``.
    """

    subunit: Specimen = field()
    rise: Union[Real_, RealVector] = field()
    twist: Union[Real_, RealVector] = field()

    pose: Pose = field(default_factory=EulerPose)

    conformations: Optional[
        Union[Conformations, Callable[[Positions], Conformations]]
    ] = field(default=None)

    n_start: int = field(static=True, default=1)
    n_subunits: Optional[int] = field(static=True, default=None)
    degrees: bool = field(static=True, default=True)

    @property
    def resolution(self) -> Real_:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution

    @cached_property
    def positions(self) -> Positions:
        """Get the helical lattice positions in the center of mass frame."""
        return compute_lattice_positions(
            self.rise,
            self.twist,
            self.subunit.pose.offset,
            n_start=self.n_start,
            n_subunits=self.n_subunits,
            degrees=self.degrees,
        )

    @cached_property
    def rotations(self) -> Rotations:
        """
        Get the helical lattice rotations in the center of mass frame.

        These are rotations of the initial subunit.
        """
        return compute_lattice_rotations(
            self.twist,
            self.subunit.pose.rotation.as_matrix(),
            n_start=self.n_start,
            n_subunits=self.n_subunits,
            degrees=self.degrees,
        )

    @cached_property
    def poses(self) -> list[Pose]:
        """Draw the poses of the subunits in the lab frame."""
        # Transform the subunit positions by pose of the helix
        transformed_positions = (
            self.pose.rotate(self.positions) + self.pose.offset
        )
        # Transform the subunit rotations by the pose of the helix
        transformed_rotations = jnp.einsum(
            "nij,jk->nik", self.rotations, self.pose.rotation.as_matrix()
        )
        # Generate a list of poses at each lattice site
        get_pose = lambda r, T: eqx.tree_at(
            lambda p: (p.offset_x, p.offset_y, p.offset_z, p.matrix),
            self.pose.as_matrix_pose(),
            (r[0, 0], r[0, 1], r[0, 2], T[0]),
        )
        poses = jtu.tree_map(
            get_pose,
            jnp.split(
                transformed_positions, self.n_start * self.n_subunits, axis=0
            ),
            jnp.split(
                transformed_rotations, self.n_start * self.n_subunits, axis=0
            ),
        )
        return poses

    @cached_property
    def subunits(self) -> list[Specimen]:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if self.conformations is None:
            subunits = self.n_subunits * self.n_start * [self.subunit]
        else:
            if isinstance(self.conformations, Callable):
                cs = self.conformations(self.positions)
            else:
                cs = self.conformations
            where = lambda s: s.conformation.coordinate
            subunits = jtu.tree_map(
                lambda c: eqx.tree_at(where, self.subunit, c[0]),
                jnp.split(cs, len(cs), axis=0),
            )
        # Assign a pose to each subunit
        get_subunit = lambda subunit, pose: eqx.tree_at(
            lambda s: s.pose, subunit, pose
        )
        subunits = jtu.tree_map(
            get_subunit,
            subunits,
            self.poses,
            is_leaf=lambda s: isinstance(s, Specimen),
        )

        return subunits


def compute_lattice_positions(
    rise: Union[Real_, RealVector],
    twist: Union[Real_, RealVector],
    initial_displacement: Float[Array, "3"],
    n_start: int = 1,
    n_subunits: Optional[int] = None,
    *,
    degrees: bool = True,
) -> Positions:
    """
    Compute the lattice points of a helix for a given
    rise, twist, radius, and start number.

    Arguments
    ---------
    rise : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical rise.
    twist : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical twist.
    initial_displacement : `Array`, shape `(3,)`
        The initial position vector of the first subunit, in
        the center of mass frame of the helix.
        The xy values are an in-plane displacement from
        the screw axis, and the z value is an offset from the
        first subunit's position.
    n_start :
        The start number of the helix.
    n_subunits :
        The number of subunits in the assembly for
        a single helix. The total number of subunits
        is really equal to ``n_start * n_subunits``.
        By default, ``2 * jnp.pi / twist``.
    degrees :
        Whether or not the angular parameters
        are given in degrees or radians.

    Returns
    -------
    lattice : shape `(n_start*n_subunits, 3)`
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

    def compute_helix_coordinates(symmetry_angle):
        """
        Get  coordinates for a given helix, where
        the x and y coordinates are rotated by an angle.
        """
        r_0 = initial_displacement
        # Define rotation about the screw axis
        c, s = jnp.cos(twist), jnp.sin(twist)
        R = jnp.array(((c, s, 0), (-s, c, 0), (0, 0, 1)), dtype=float)

        # Coordinate transformation between subunits
        def f(carry, x):
            y = R @ carry + jnp.asarray((0, 0, rise), dtype=float)
            return y, y

        _, r = jax.lax.scan(f, r_0, None, length=n_subunits - 1)
        r = jnp.insert(r, 0, r_0, axis=0)
        # Shift helix center of mass to the origin
        r -= jnp.asarray([0.0, 0.0, rise * n_subunits / 2], dtype=float)
        # Transformation between helical strands from start-number
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(
            ((c_n, s_n, 0), (-s_n, c_n, 0), (0, 0, 1)), dtype=float
        )
        return (R_n @ r.T).T

    # The helical coordinates for all sub-helices
    positions = jax.vmap(compute_helix_coordinates)(symmetry_angles)
    return positions.reshape((n_start * n_subunits, 3))


def compute_lattice_rotations(
    twist: Union[Real_, RealVector],
    initial_rotation: Float[Array, "3 3"],
    n_start: int = 1,
    n_subunits: Optional[int] = None,
    *,
    degrees: bool = True,
) -> Rotations:
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

    def compute_helix_rotations(symmetry_angle):
        """
        Get  coordinates for a given helix, where
        the x and y coordinates are rotated by an angle.
        """
        T_0 = initial_rotation
        # Define rotation about the screw axis
        c, s = jnp.cos(twist), jnp.sin(twist)
        R = jnp.array(((c, s, 0), (-s, c, 0), (0, 0, 1)), dtype=float)

        # Coordinate transformation between subunits
        def f(carry, x):
            y = R.T @ carry
            return y, y

        _, T = jax.lax.scan(f, T_0, None, length=n_subunits - 1)
        T = jnp.insert(T, 0, T_0, axis=0)
        # Transformation between helical strands from start-number
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(
            ((c_n, s_n, 0), (-s_n, c_n, 0), (0, 0, 1)), dtype=float
        )
        return jnp.einsum("ij,njk->nik", R_n, T)

    rotations = jax.vmap(compute_helix_rotations)(symmetry_angles)
    return rotations.reshape((n_start * n_subunits, 3, 3))

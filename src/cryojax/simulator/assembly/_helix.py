"""
Abstraction of a helical polymer.
"""

from __future__ import annotations

__all__ = ["Helix", "compute_lattice_positions", "compute_lattice_rotations"]

from typing import Union, Optional, Any
from jaxtyping import Array, Float
from functools import cached_property

import jax
import jax.numpy as jnp

from ..specimen import Ensemble
from ._assembly import Assembly, _Positions, _Rotations

from ...core import field
from ...typing import Real_, RealVector

_RotationMatrix3D = Float[Array, "3 3"]

_Vector3D = Float[Array, "3"]


class Helix(Assembly):
    """
    Abstraction of a helical polymer.

    This class assembles a helix from a subunit.
    See the ``Assembly`` base class for more information.

    The screw axis is taken to be in the center of the
    image, pointing out-of-plane (i.e. along the z direction).

    Attributes
    ----------
    subunit :
        The helical subunit. It is important to set the the initial pose
        of the initial subunit here. This is in the center of mass frame
        of the helix with the screw axis pointing in the z-direction.
    rise :
        The helical rise. This has dimensions of length.
    twist :
        The helical twist, given in degrees if
        ``degrees = True`` and radians otherwise.
    pose :
        The center of mass pose of the helix.
    conformations :
        The conformation of `subunit` at each lattice site.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the lattice positions.
        In either case, the `Array` should be shape `(n_start*n_subunits_per_start,)`.
    n_start :
        The start number of the helix. By default, ``1``.
    n_subunits_per_start :
        The number of subunits each helical strand.
        By default, ``1``.
    degrees :
        Whether or not the helical twist is given in
        degrees. By default, ``True``.
    """

    rise: Union[Real_, RealVector] = field()
    twist: Union[Real_, RealVector] = field()

    n_start: int = field(static=True)
    n_subunits_per_start: int = field(static=True)
    degrees: bool = field(static=True)

    def __init__(
        self,
        subunit: Ensemble,
        rise: Union[Real_, RealVector],
        twist: Union[Real_, RealVector],
        *,
        n_start: int = 1,
        n_subunits_per_start: int = 1,
        degrees: bool = True,
        **kwargs: Any,
    ):
        super().__init__(subunit, **kwargs)
        self.rise = rise
        self.twist = twist
        self.n_start = n_start
        self.n_subunits_per_start = n_subunits_per_start
        self.degrees = degrees

    @cached_property
    def n_subunits(self) -> int:
        """The number of subunits in the assembly"""
        return self.n_start * self.n_subunits_per_start

    @cached_property
    def positions(self) -> _Positions:
        """Get the helical lattice positions in the center of mass frame."""
        return compute_lattice_positions(
            self.rise,
            self.twist,
            self.subunit.pose.offset,
            n_start=self.n_start,
            n_subunits_per_start=self.n_subunits_per_start,
            degrees=self.degrees,
        )

    @cached_property
    def rotations(self) -> _Rotations:
        """
        Get the helical lattice rotations in the center of mass frame.

        These are rotations of the initial subunit.
        """
        return compute_lattice_rotations(
            self.twist,
            self.subunit.pose.rotation.as_matrix(),
            n_start=self.n_start,
            n_subunits_per_start=self.n_subunits_per_start,
            degrees=self.degrees,
        )


def compute_lattice_positions(
    rise: Union[Real_, RealVector],
    twist: Union[Real_, RealVector],
    initial_displacement: _Vector3D,
    n_start: int = 1,
    n_subunits_per_start: Optional[int] = None,
    *,
    degrees: bool = True,
) -> _Positions:
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
    n_subunits_per_start :
        The number of subunits in the assembly for
        a single helix. The total number of subunits
        is really equal to ``n_start * n_subunits``.
        By default, ``2 * jnp.pi / twist``.
    degrees :
        Whether or not the angular parameters
        are given in degrees or radians.

    Returns
    -------
    positions : shape `(n_start*n_subunits_per_start, 3)`
        The helical lattice positions.
    """
    # Convert to radians
    if degrees:
        twist = jnp.deg2rad(twist)
    # If the number of subunits is not given, compute for one helix
    if n_subunits_per_start is None:
        n_subunits_per_start = abs(int(2 * jnp.pi / twist))
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
            y = R.T @ carry + jnp.asarray((0, 0, rise), dtype=float)
            return y, y

        _, r = jax.lax.scan(f, r_0, None, length=n_subunits_per_start - 1)
        r = jnp.insert(r, 0, r_0, axis=0)
        # Shift helix center of mass to the origin
        delta_z = rise * jnp.asarray(n_subunits_per_start - 1, dtype=float) / 2
        r -= jnp.asarray((0.0, 0.0, delta_z), dtype=float)
        # Transformation between helical strands from start-number
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(
            ((c_n, s_n, 0), (-s_n, c_n, 0), (0, 0, 1)), dtype=float
        )
        return (R_n.T @ r.T).T

    # The helical coordinates for all sub-helices
    positions = jax.vmap(compute_helix_coordinates)(symmetry_angles)
    return positions.reshape((n_start * n_subunits_per_start, 3))


def compute_lattice_rotations(
    twist: Union[Real_, RealVector],
    initial_rotation: _RotationMatrix3D,
    n_start: int = 1,
    n_subunits_per_start: Optional[int] = None,
    *,
    degrees: bool = True,
) -> _Rotations:
    """
    Compute the relative rotations of subunits on a
    helical lattice, parameterized by the
    rise, twist, start number, and an initial rotation.

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
    n_subunits_per_start :
        The number of subunits in the assembly for
        a single helix. The total number of subunits
        is really equal to ``n_start * n_subunits``.
        By default, ``2 * jnp.pi / twist``.
    degrees :
        Whether or not the angular parameters
        are given in degrees or radians.

    Returns
    -------
    rotations : shape `(n_start*n_subunits_per_start, 3, 3)`
        The relative rotations between subunits
        on the helical lattice.
    """
    # Convert to radians
    if degrees:
        twist = jnp.deg2rad(twist)
    # If the number of subunits is not given, compute for one helix
    if n_subunits_per_start is None:
        n_subunits_per_start = int(2 * jnp.pi / twist)
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
            y = R @ carry
            return y, y

        _, T = jax.lax.scan(f, T_0, None, length=n_subunits_per_start - 1)
        T = jnp.insert(T, 0, T_0, axis=0)
        # Transformation between helical strands from start-number
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(
            ((c_n, s_n, 0), (-s_n, c_n, 0), (0, 0, 1)), dtype=float
        )
        return jnp.einsum("ij,njk->nik", R_n, T)

    rotations = jax.vmap(compute_helix_rotations)(symmetry_angles)
    return rotations.reshape((n_start * n_subunits_per_start, 3, 3))

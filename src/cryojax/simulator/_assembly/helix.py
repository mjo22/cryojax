"""
Abstraction of a helical polymer.
"""

from functools import cached_property
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...rotations import SO3
from .._pose import AbstractPose, EulerAnglePose
from .._structural_ensemble import (
    AbstractConformationalVariable,
    AbstractStructuralEnsemble,
)
from .assembly import AbstractAssembly


class HelicalAssembly(AbstractAssembly, strict=True):
    """
    Abstraction of a helical polymer.

    This class assembles a helix from a subunit.
    See the ``AbstractAssembly`` base class for more information.

    The screw axis is taken to be in the center of the
    image, pointing out-of-plane (i.e. along the z direction).
    """

    subunit: AbstractStructuralEnsemble
    rise: Float[Array, ""]
    twist: Float[Array, ""]

    pose: AbstractPose
    conformation: Optional[AbstractConformationalVariable]

    n_subunits: int
    n_start: int

    def __init__(
        self,
        subunit: AbstractStructuralEnsemble,
        rise: Float[Array, ""] | float,
        twist: Float[Array, ""] | float,
        pose: Optional[AbstractPose] = None,
        conformation: Optional[AbstractConformationalVariable] = None,
        n_start: int = 1,
        n_subunits: int = 1,
    ):
        """**Arguments:**
        - `subunit`:
            The helical subunit. It is important to set the the initial pose
            of the initial subunit here. This is in the center of mass frame
            of the helix with the screw axis pointing in the z-direction.
        - `rise`:
            The helical rise. This has dimensions of length.
        - `twist`:
            The helical twist.
        - `pose`:
            The center of mass pose of the helix.
        - `conformation`:
            The conformation of `subunit` at each lattice site.
            This can either be a fixed set of conformations or a function
            that computes conformations based on the lattice positions.
            In either case, the `Array` should be shape
            `(n_start*n_subunits_per_start,)`.
        - `n_start`:
            The start number of the helix. By default, ``1``.
        - `n_subunits_per_start`:
            The number of subunits each helical strand.
            By default, ``1``.
        """
        self.subunit = subunit
        self.pose = pose or EulerAnglePose()
        self.rise = jnp.asarray(rise)
        self.twist = jnp.asarray(twist)
        self.conformation = conformation
        self.n_start = n_start
        self.n_subunits = n_subunits

    def __check_init__(self):
        if self.n_subunits % self.n_start != 0:
            raise AttributeError(
                "The number of subunits must be a multiple of the helical start number."
            )

    @cached_property
    def offsets_in_angstroms(self) -> Float[Array, "{self.n_subunits} 3"]:
        """Get the helical lattice positions in the center of mass frame."""
        return compute_helical_lattice_positions(
            self.rise,
            self.twist,
            n_subunits_per_start=self.n_subunits // self.n_start,
            initial_displacement=self.subunit.pose.offset_in_angstroms,
            n_start=self.n_start,
        )

    @cached_property
    def rotations(self) -> SO3:
        """Get the helical lattice rotations in the center of mass frame.

        These are rotations of the initial subunit.
        """
        transformed_rotation_matrices = compute_helical_lattice_rotations(
            self.twist,
            n_subunits_per_start=self.n_subunits // self.n_start,
            initial_rotation=self.subunit.pose.rotation.as_matrix(),
            n_start=self.n_start,
        )
        # Function to construct SO3 objects vmapped over leading dimension
        transformed_rotations = jax.vmap(lambda mat: SO3.from_matrix(mat))(
            transformed_rotation_matrices
        )
        return transformed_rotations


def compute_helical_lattice_positions(
    rise: Float[Array, ""],
    twist: Float[Array, ""],
    n_subunits_per_start: int,
    initial_displacement: Float[Array, "3"],
    n_start: int = 1,
) -> Float[Array, "{n_start*n_subunits_per_start} 3"]:
    """
    Compute the lattice points of a helix for a given
    rise, twist, radius, and start number.

    Arguments
    ---------
    rise : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical rise.
    twist : `Real_` or `RealVector`, shape `(n_subunits,)`
        The helical twist.
    n_subunits_per_start :
        The number of subunits in the assembly for
        a single sub-helix.
    initial_displacement : `Array`, shape `(3,)`
        The initial position vector of the first subunit, in
        the center of mass frame of the helix.
        The xy values are an in-plane displacement from
        the screw axis, and the z value is an offset from the
        first subunit's position.
    n_start :
        The start number of the helix.

    Returns
    -------
    subunit_positions : shape `(n_start*n_subunits_per_start, 3)`
        The helical lattice positions.
    """
    # Convert to radians
    twist = jnp.deg2rad(twist)

    # Coordinate transformation to get subunit positions in a single sub-helix
    def compute_ith_subunit_position_in_subhelix(i, theta, dz, r_0, N):
        # ... define rotation about the screw axis
        c, s = jnp.cos(i * theta), jnp.sin(i * theta)
        R_t = jnp.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)
        # ... transform by i rises and i twists
        r_i = R_t @ r_0 + i * jnp.asarray((0, 0, dz), dtype=float)
        # ... center positions of subunits in z
        return r_i - jnp.asarray(
            (0.0, 0.0, dz * jnp.asarray(N - 1, dtype=float) / 2), dtype=float
        )

    # ... function to get positions of all subunits
    compute_subunit_positions_in_subhelix = jax.vmap(
        compute_ith_subunit_position_in_subhelix,
        in_axes=[0, None, None, None, None],
    )
    # ... get indices of subunits along sub-helix
    subunit_indices = jnp.arange(n_subunits_per_start, dtype=float)
    # ... compute positions of subunits in sub-helix
    subunit_positions_in_subhelix = compute_subunit_positions_in_subhelix(
        subunit_indices,
        twist,
        rise,
        initial_displacement,
        n_subunits_per_start,
    )

    # Now, transform this single sub-helix into all sub-helices, related by C_n
    # symmetry
    def compute_helix_subunit_positions_per_start(symmetry_angle, r):
        # ... rotate the sub-helix around the screw axis to a different sub-helix
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(((c_n, -s_n, 0), (s_n, c_n, 0), (0, 0, 1)), dtype=float)
        return (R_n @ r.T).T

    # ... function to rotate entire sub-helix around the screw axis
    compute_helix_subunit_positions = jax.vmap(
        compute_helix_subunit_positions_per_start, in_axes=[0, None]
    )
    # ... compute symmetry angles relating first sub-helix to all other sub-helices
    symmetry_angles = jnp.array([2 * jnp.pi * n / n_start for n in range(n_start)])
    # ... finally, get all subunit positions!
    subunit_positions = compute_helix_subunit_positions(
        symmetry_angles, subunit_positions_in_subhelix
    ).reshape((n_start * n_subunits_per_start, 3))

    return subunit_positions


def compute_helical_lattice_rotations(
    twist: Float[Array, ""],
    n_subunits_per_start: int,
    initial_rotation: Float[Array, "3 3"] = jnp.eye(3),
    n_start: int = 1,
) -> Float[Array, "{n_start*n_subunits_per_start} 3 3"]:
    """
    Compute the relative rotations of subunits on a
    helical lattice, parameterized by the
    rise, twist, start number, and an initial rotation.

    Arguments
    ---------
    twist : `Real_`
        The helical twist.
    n_subunits_per_start :
        The number of subunits in the assembly for
        a single sub-helix.
    initial_rotation : `Array`, shape `(3, 3)`
        The initial rotation of the first subunit. By default,
        the identity matrix.
    n_start :
        The start number of the helix.

    Returns
    -------
    subunit_rotations : shape `(n_start*n_subunits_per_start, 3, 3)`
        The relative rotations between subunits
        on the helical lattice.
    """
    # Convert to radians
    twist = jnp.deg2rad(twist)

    # Coordinate transformation to get subunit rotations in a single sub-helix
    def compute_ith_subunit_rotation_in_subhelix(i, theta, R_0):
        # ... define rotation about the screw axis
        c, s = jnp.cos(i * theta), jnp.sin(i * theta)
        R_t = jnp.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)
        # ... transform to the ith position
        return R_t @ R_0

    # ... function to get rotations of all subunits
    compute_subunit_rotations_in_subhelix = jax.vmap(
        compute_ith_subunit_rotation_in_subhelix, in_axes=[0, None, None]
    )
    # ... get indices of subunits along sub-helix
    subunit_indices = jnp.arange(n_subunits_per_start, dtype=float)
    # ... compute rotations of subunits in sub-helix
    subunit_rotations_in_subhelix = compute_subunit_rotations_in_subhelix(
        subunit_indices, twist, initial_rotation
    )

    # Now, transform single sub-helix rotations into all sub-helices rotations,
    # related by C_n symmetry
    def compute_helix_subunit_rotations_per_start(symmetry_angle, R):
        # ... rotate the sub-helix around the screw axis to a different sub-helix
        c_n, s_n = jnp.cos(symmetry_angle), jnp.sin(symmetry_angle)
        R_n = jnp.array(((c_n, -s_n, 0), (s_n, c_n, 0), (0, 0, 1)), dtype=float)
        return jnp.einsum("ij,njk->nik", R_n, R)

    # ... function to rotate subunit about symmetry angle
    compute_helix_subunit_rotations = jax.vmap(
        compute_helix_subunit_rotations_per_start, in_axes=[0, None]
    )
    # ... compute symmetry angles relating first sub-helix to all other sub-helices
    symmetry_angles = jnp.array([2 * jnp.pi * n / n_start for n in range(n_start)])
    # ... finally, get all subunit rotations!
    subunit_rotations = compute_helix_subunit_rotations(
        symmetry_angles, subunit_rotations_in_subhelix
    ).reshape((n_start * n_subunits_per_start, 3, 3))

    return subunit_rotations

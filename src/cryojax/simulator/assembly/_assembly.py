"""
Abstraction of a biological assembly. This assembles a structure
by adding together a collection of subunits, parameterized by
some geometry.
"""

from __future__ import annotations

__all__ = ["Assembly"]

from abc import abstractmethod
from typing import Union, Optional, Callable
from jaxtyping import Array, Float, Int
from functools import cached_property

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ..specimen import Specimen
from ..pose import Pose, EulerPose

from ...core import field, Module
from ...typing import Real_, Int_

_Positions = Float[Array, "N 3"]
"""Type hint for array where each element is a subunit coordinate."""

_Rotations = Float[Array, "N 3 3"]
"""Type hint for array where each element is a subunit rotation."""

_Conformations = Int[Array, "N"]
"""Type hint for array where each element updates a Conformation."""


class Assembly(Module):
    """
    Abstraction of a biological assembly.

    This class acts just like a ``Specimen``, however
    it creates an assembly from a subunit.

    To subclass an ``Assembly``,
        1) Overwrite the ``Assembly.n_subunits``
           property
        2) Overwrite the ``Assembly.positions``
           and ``Assembly.rotations`` properties.

    Attributes
    ----------
    subunit :
        The subunit. It is important to set the the initial pose
        of the initial subunit here. The initial pose is not in
        the lab frame, it is in the center of mass frame of the assembly.
    pose :
        The center of mass pose of the helix.
    conformations :
        The conformation of each `subunit`.
        This can either be a fixed set of conformations or a function
        that computes conformations based on the subunit positions.
        In either case, the `Array` should be shape `(n_start*n_subunits,)`.
    """

    subunit: Specimen = field()
    pose: Pose = field(default_factory=EulerPose)

    conformations: Optional[
        Union[_Conformations, Callable[[_Positions], _Conformations]]
    ] = field(default=None)

    def __init__(
        self,
        subunit: Specimen,
        *,
        pose: Optional[Pose] = None,
        conformations: Optional[
            Union[_Conformations, Callable[[_Positions], _Conformations]]
        ] = None,
    ):
        self.subunit = subunit
        self.pose = pose or EulerPose()
        self.conformations = conformations

    @property
    def resolution(self) -> Real_:
        """Hack to make this class act like a Specimen."""
        return self.subunit.resolution

    @cached_property
    @abstractmethod
    def n_subunits(self) -> int:
        """The number of subunits in the assembly"""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def positions(self) -> _Positions:
        """The positions of each subunit."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rotations(self) -> _Rotations:
        """
        The relative rotations between subunits
        """
        raise NotImplementedError

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
            jnp.split(transformed_positions, self.n_subunits, axis=0),
            jnp.split(transformed_rotations, self.n_subunits, axis=0),
        )
        return poses

    @cached_property
    def subunits(self) -> list[Specimen]:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if self.conformations is None:
            subunits = [self.subunit for _ in range(self.n_subunits)]
        else:
            if isinstance(self.conformations, Callable):
                cs = self.conformations(self.positions)
            else:
                cs = self.conformations
            where = lambda s: s.conformation.coordinate
            subunits = jtu.tree_map(
                lambda c: eqx.tree_at(where, self.subunit, c[0]),
                jnp.split(cs, self.n_subunits, axis=0),
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

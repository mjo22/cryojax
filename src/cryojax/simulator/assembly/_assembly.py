"""
Abstraction of a biological assembly. This assembles a structure
by adding together a collection of subunits, parameterized by
some geometry.
"""

from __future__ import annotations

__all__ = ["Assembly"]

from abc import abstractmethod
from typing import Optional, Callable
from jaxtyping import Array, Float, Int
from functools import cached_property

import jax.numpy as jnp
import equinox as eqx

from ..specimen import Specimen
from ..pose import Pose, EulerPose, MatrixPose

from ...core import field, Module
from ...typing import Real_

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
    conformation_fn :
        A function that computes conformations based on the subunit positions.
    """

    subunit: Specimen = field()
    pose: Pose = field()
    conformations: Optional[_Conformations] = field()
    conformation_fn: Optional[Callable[[_Positions], _Conformations]] = field(
        static=True
    )

    def __init__(
        self,
        subunit: Specimen,
        *,
        pose: Optional[Pose] = None,
        conformations: Optional[_Conformations] = None,
        conformation_fn: Optional[
            Callable[[_Positions], _Conformations]
        ] = None,
    ):
        self.subunit = subunit
        self.pose = pose or EulerPose()
        self.conformations = conformations
        self.conformation_fn = conformation_fn

    def __check_init__(self):
        if self.conformations is not None and self.conformation_fn is not None:
            raise ValueError(
                "Only one of Assembly.conformations or Assembly.conformation_fn should be set."
            )

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
        """The relative rotations between subunits."""
        raise NotImplementedError

    @cached_property
    def poses(self) -> Pose:
        """
        Draw the poses of the subunits in the lab frame, measured
        from the rotation relative to the first subunit.
        """
        # Transform the subunit positions by pose of the helix
        transformed_positions = (
            self.pose.rotate(self.positions) + self.pose.offset
        )
        # Transform the subunit rotations by the pose of the helix
        transformed_rotations = jnp.einsum(
            "nij,jk->nik", self.rotations, self.pose.rotation.as_matrix()
        )
        return MatrixPose(
            offset_x=transformed_positions[:, 0],
            offset_y=transformed_positions[:, 1],
            offset_z=transformed_positions[:, 2],
            matrix=transformed_rotations,
        )

    @cached_property
    def subunits(self) -> Specimen:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if [self.conformations, self.conformation_fn] == [None, None]:
            where = lambda s: s.pose
            return eqx.tree_at(where, self.subunit, self.poses)
        else:
            if self.conformation_fn is not None:
                cs = self.conformation_fn(self.positions)
            else:
                cs = self.conformations
            where = lambda s: (s.conformation.coordinate, s.pose)
            return eqx.tree_at(where, self.subunit, (cs, self.poses))

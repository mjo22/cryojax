"""
Abstraction of a biological assembly. This assembles a structure
by computing an Ensemble of subunits, parameterized by
some geometry.
"""

from __future__ import annotations

__all__ = ["AbstractAssembly"]

from abc import abstractmethod
from typing import Optional
from jaxtyping import Array, Float
from functools import cached_property

import jax.numpy as jnp
import equinox as eqx

from ..specimen import AbstractSpecimen, AbstractEnsemble
from ..conformation import AbstractConformation
from ..pose import AbstractPose, EulerPose, MatrixPose

_Positions = Float[Array, "N 3"]
"""Type hint for array where each element is a subunit coordinate."""

_Rotations = Float[Array, "N 3 3"]
"""Type hint for array where each element is a subunit rotation."""


class AbstractAssembly(eqx.Module):
    """
    Abstraction of a biological assembly.

    This class acts just like an ``AbstractSpecimen``, however
    it creates an assembly from a subunit.

    To subclass an ``AbstractAssembly``,
        1) Overwrite the ``AbstractAssembly.n_subunits``
           property
        2) Overwrite the ``AbstractAssembly.positions``
           and ``AbstractAssembly.rotations`` properties.

    Attributes
    ----------
    subunit :
        The subunit. It is important to set the the initial pose
        of the initial subunit here. The initial pose is not in
        the lab frame, it is in the center of mass frame of the assembly.
    pose :
        The center of mass pose of the helix.
    conformation :
        The conformation of each `subunit`.
    """

    subunit: AbstractSpecimen
    pose: AbstractPose
    conformation: Optional[AbstractConformation] = None

    def __init__(
        self,
        subunit: AbstractSpecimen,
        *,
        pose: Optional[AbstractPose] = None,
        conformation: Optional[AbstractConformation] = None,
    ):
        self.subunit = subunit
        self.pose = pose or EulerPose()
        self.conformation = conformation
        # Make sure that if conformation is set, subunit is an Ensemble
        if conformation is not None and not isinstance(
            subunit, AbstractEnsemble
        ):
            cls = type(self)
            raise AttributeError(
                f"If {cls}.conformation is set, {cls}.subunit must be an Ensemble."
            )

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
    def poses(self) -> AbstractPose:
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
    def subunits(self) -> AbstractSpecimen:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if isinstance(self.subunit, AbstractEnsemble):
            where = lambda s: (s.conformation, s.pose)
            return eqx.tree_at(
                where, self.subunit, (self.conformation, self.poses)
            )
        else:
            where = lambda s: s.pose
            return eqx.tree_at(where, self.subunit, self.poses)

"""
Abstraction of a biological assembly. This assembles a structure
by computing an Ensemble of subunits, parameterized by
some geometry.
"""

from abc import abstractmethod
from typing import Optional
from jaxtyping import Array, Float
from functools import cached_property
from equinox import AbstractVar

import jax
import jax.numpy as jnp
import equinox as eqx

from .._specimen import AbstractSpecimen, AbstractEnsemble
from .._conformation import AbstractConformation
from .._pose import AbstractPose, MatrixPose


class AbstractAssembly(eqx.Module, strict=True):
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

    subunit: AbstractVar[AbstractSpecimen]
    pose: AbstractVar[AbstractPose]
    conformation: AbstractVar[Optional[AbstractConformation]]

    n_subunits: AbstractVar[int]

    def __check_init__(self):
        if self.conformation is not None and not isinstance(
            self.subunit, AbstractEnsemble
        ):
            # Make sure that if conformation is set, subunit is an AbstractEnsemble
            raise AttributeError(
                f"If {type(self)}.conformation is set, {type(self)}.subunit must be an AbstractEnsemble."
            )
        if self.conformation is not None and isinstance(
            self.subunit, AbstractEnsemble
        ):
            # ... if it is an AbstractEnsemble, the AbstractConformation must be the right type
            if not isinstance(
                self.conformation, type(self.subunit.conformation)
            ):
                raise AttributeError(
                    f"{type(self)}.conformation must be type {type(self.subunit.conformation)} if {type(self)}.subunit is type {type(self.subunit)}."
                )

    @cached_property
    @abstractmethod
    def positions(self) -> Float[Array, "n_subunits 3"]:
        """The positions of each subunit."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rotations(self) -> Float[Array, "n_subunits 3 3"]:
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
            self.pose.rotate_coordinates(self.positions, inverse=False)
            + self.pose.offset
        )
        # Transform the subunit rotations by the pose of the helix. This operation
        # left multiplies by the pose of the helix, taking care that first subunits
        # are rotated to the center of mass frame, then the lab frame.
        transformed_rotations = jnp.einsum(
            "ij,njk->nik", self.pose.rotation.as_matrix(), self.rotations
        )
        # Function to construct a MatrixPose, vmapped over leading dimension
        make_assembly_poses = jax.vmap(
            lambda pos, rot: MatrixPose(
                offset_x=pos[0], offset_y=pos[1], offset_z=pos[2], matrix=rot
            )
        )

        return make_assembly_poses(
            transformed_positions, transformed_rotations
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

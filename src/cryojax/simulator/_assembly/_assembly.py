"""
Abstraction of a biological assembly. This assembles a structure
by computing an Ensemble of subunits, parameterized by
some geometry.
"""

from abc import abstractmethod
from functools import cached_property
from typing import Optional

import equinox as eqx
import jax
from equinox import AbstractVar
from jaxtyping import Array, Float, Shaped

from ...rotations import SO3
from .._conformation import AbstractConformation
from .._pose import AbstractPose
from .._specimen import AbstractEnsemble, AbstractSpecimen


class AbstractAssembly(eqx.Module, strict=True):
    """Abstraction of a biological assembly.

    This class acts just like an ``AbstractSpecimen``, however
    it creates an assembly from a subunit.

    To subclass an `AbstractAssembly`,
        1) Overwrite the `AbstractAssembly.n_subunits`
           property
        2) Overwrite the `AbstractAssembly.positions`
           and `AbstractAssembly.rotations` properties.
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
                f"If {type(self)}.conformation is set, {type(self)}.subunit must be an "
                "AbstractEnsemble."
            )
        if self.conformation is not None and isinstance(self.subunit, AbstractEnsemble):
            # ... if it is an AbstractEnsemble, the AbstractConformation must be the
            #  right type
            if not isinstance(self.conformation, type(self.subunit.conformation)):
                raise AttributeError(
                    f"{type(self)}.conformation must be type "
                    f" {type(self.subunit.conformation)} if {type(self)}.subunit is "
                    f"type {type(self.subunit)}."
                )

    @cached_property
    @abstractmethod
    def offsets_in_angstroms(self) -> Float[Array, "n_subunits 3"]:
        """The positions of each subunit."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rotations(self) -> Shaped[SO3, " n_subunits"]:
        """The relative rotations between subunits."""
        raise NotImplementedError

    @cached_property
    def poses(self) -> Shaped[AbstractPose, " n_subunits"]:
        """
        Draw the poses of the subunits in the lab frame, measured
        from the rotation relative to the first subunit.
        """
        # Transform the subunit positions by pose of the helix
        transformed_positions = (
            self.pose.rotate_coordinates(self.offsets_in_angstroms, inverse=False)
            + self.pose.offset_in_angstroms
        )
        # Transform the subunit rotations by the pose of the helix. This operation
        # left multiplies by the pose of the helix, taking care that first subunits
        # are rotated to the center of mass frame, then the lab frame.
        transformed_rotations = jax.vmap(
            lambda com_rotation, subunit_rotation: com_rotation @ subunit_rotation,
            in_axes=[None, 0],
        )(self.pose.rotation, self.rotations)
        # Function to construct AbstractPoses
        cls = type(self.pose)
        make_assembly_poses = jax.vmap(
            lambda rot, pos: cls.from_rotation_and_translation(rot, pos)
        )

        return make_assembly_poses(transformed_rotations, transformed_positions)

    @cached_property
    def subunits(self) -> AbstractSpecimen:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if isinstance(self.subunit, AbstractEnsemble):
            where = lambda s: (s.conformation, s.pose)
            return eqx.tree_at(where, self.subunit, (self.conformation, self.poses))
        else:
            where = lambda s: s.pose
            return eqx.tree_at(where, self.subunit, self.poses)

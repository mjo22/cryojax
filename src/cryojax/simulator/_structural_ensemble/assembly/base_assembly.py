"""
Abstraction of a biological assembly. This assembles a structure
by computing a batch of subunits, parameterized by some geometry.
"""

from abc import abstractmethod
from functools import cached_property
from typing_extensions import override

import equinox as eqx
import jax
from equinox import AbstractVar
from jaxtyping import Array, Float

from ....rotations import SO3
from ..._pose import AbstractPose
from ..._potential_representation import AbstractPotentialRepresentation
from ..base_ensemble import AbstractStructuralEnsemble


class AbstractAssembly(AbstractStructuralEnsemble, strict=True):
    """Abstraction of a biological assembly.

    To subclass an `AbstractAssembly`,
        1) Overwrite the `AbstractAssembly.n_subunits`
           property
        2) Overwrite the `AbstractAssembly.offsets_in_angstroms`
           and `AbstractAssembly.rotations` properties.
    """

    @override
    def get_potential_at_conformation(self) -> AbstractPotentialRepresentation:
        raise NotImplementedError(
            "Method to construct a potential from an "
            "`AbstractAssembly` concrete class not yet supported."
        )

    @abstractmethod
    def get_subcomponents(self) -> AbstractStructuralEnsemble:
        """Get the subcomponents of the assembly, represented
        as an `AbstractStructuralEnsemble` where each entry has
        a batch dimension.
        """
        raise NotImplementedError


class AbstractAssemblyWithSubunit(AbstractAssembly, strict=True):
    """Abstraction of a biological assembly with a single
    assymmetric subunit (ASU).
    """

    subunit: AbstractVar[AbstractStructuralEnsemble]
    n_subunits: AbstractVar[int]

    def __check_init__(self):
        if self.conformation is not None and self.subunit.conformation is None:
            # Make sure that if conformation is set, subunit is an AbstractEnsemble
            raise AttributeError(
                f"If {type(self)}.conformation is set, "
                "{type(self)}.subunit.conformation cannot be `None`."
            )
        if self.conformation is not None and self.subunit.conformation is not None:
            # ... if it is an AbstractEnsemble, the AbstractConformation must be the
            #  right type
            if not isinstance(self.conformation, type(self.subunit.conformation)):
                raise AttributeError(
                    f"{type(self)}.conformation must be type "
                    f" {type(self.subunit.conformation)} if {type(self)}.subunit is "
                    f"type {type(self.subunit)}."
                )

    @override
    def get_subcomponents(self) -> AbstractStructuralEnsemble:
        return self.subunits

    @cached_property
    @abstractmethod
    def offsets_in_angstroms(self) -> Float[Array, "{n_subunits} 3"]:
        """The 3D positions of each subunit."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rotations(self) -> SO3:
        """The relative rotations between subunits."""
        raise NotImplementedError

    @cached_property
    def poses(self) -> AbstractPose:
        """
        Draw the poses of the subunits in the lab frame, measured
        from the rotation relative to the first subunit.
        """
        # Transform the subunit positions by the center of mass pose of the assembly.
        transformed_positions = (
            self.pose.rotate_coordinates(self.offsets_in_angstroms, inverse=False)
            + self.pose.offset_in_angstroms
        )
        # Transform the subunit rotations by the center of mass pose of the assembly.
        # This operation left multiplies by the pose rotation matrix, taking care that
        # first subunits are rotated to the center of mass frame, then the lab frame.
        transformed_rotations = jax.vmap(
            lambda com_rotation, subunit_rotation: com_rotation @ subunit_rotation,
            in_axes=[None, 0],
        )(self.pose.rotation, self.rotations)
        # Construct the batch of `AbstractPose`s
        cls = type(self.pose)
        make_assembly_poses = jax.vmap(
            lambda rot, pos: cls.from_rotation_and_translation(rot, pos)
        )

        return make_assembly_poses(transformed_rotations, transformed_positions)

    @cached_property
    def subunits(self) -> AbstractStructuralEnsemble:
        """Draw a realization of all of the subunits in the lab frame."""
        # Compute a list of subunits, configured at the correct conformations
        if self.subunit.conformation is not None:
            where = lambda s: (s.conformation, s.pose)
            return eqx.tree_at(where, self.subunit, (self.conformation, self.poses))
        else:
            where = lambda s: s.pose
            return eqx.tree_at(where, self.subunit, self.poses)

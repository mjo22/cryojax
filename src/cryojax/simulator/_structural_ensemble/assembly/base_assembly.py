"""
Abstraction of a biological assembly. This assembles a structure
by computing a batch of subunits, parameterized by some geometry.
"""

from abc import abstractmethod
from functools import cached_property
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
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

    n_subcomponents: AbstractVar[int]

    @override
    def get_potential_in_body_frame(self) -> AbstractPotentialRepresentation:
        raise NotImplementedError(
            "Method to construct a potential from an "
            "`AbstractAssembly` concrete class not yet supported."
        )

    @abstractmethod
    def get_subcomponents_and_z_positions_in_body_frame(
        self,
    ) -> tuple[AbstractStructuralEnsemble, Float[Array, " {self.n_subcomponents}"]]:
        """Get the subcomponents of the assembly, represented
        as an `AbstractStructuralEnsemble` where each entry has
        a batch dimension. Also, return the z-position of each
        subcomponent.
        """
        raise NotImplementedError

    @abstractmethod
    def get_subcomponents_and_z_positions_in_lab_frame(
        self,
    ) -> tuple[AbstractStructuralEnsemble, Float[Array, " {self.n_subcomponents}"]]:
        """Get the subcomponents of the assembly in the lab frame,
        represented as an `AbstractStructuralEnsemble` where each
        entry has a batch dimension. Also, return the z-position of
        each subcomponent.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def positions_in_body_frame(self) -> Float[Array, "{self.n_subcomponents} 3"]:
        """The 3D positions of each subcomponent, measured
        in angstroms and relative to the center of the volume."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rotations_in_body_frame(self) -> SO3:
        """The relative rotations between subcomponents."""
        raise NotImplementedError

    @cached_property
    def positions_in_lab_frame(
        self,
    ) -> Float[Array, "{self.n_subcomponents} 3"]:
        """The 3D positions of each subcomponent in the lab frame,
        measured in angstroms and relative to the center of the volume."""
        return (
            self.pose.rotate_coordinates(self.positions_in_body_frame, inverse=False)
            + jnp.concatenate((self.pose.offset_in_angstroms, jnp.zeros(1)))[None, :]
        )

    @cached_property
    def rotations_in_lab_frame(self) -> SO3:
        """The relative rotations between subcomponents in the lab frame.

        This operation left multiplies by the pose rotation matrix, taking care that
        first subcomponents are rotated to the center of mass frame, then the lab frame.
        """
        rotate_into_lab_frame = jax.vmap(
            lambda lab_frame_rotation, subcomponent_rotation: lab_frame_rotation
            @ subcomponent_rotation,
            in_axes=[None, 0],
        )
        return rotate_into_lab_frame(self.pose.rotation, self.rotations_in_body_frame)

    def get_poses_and_z_positions_in_body_frame(
        self,
    ) -> tuple[AbstractPose, Float[Array, " {self.n_subcomponents}"]]:
        """Draw the poses of the subcomponents in the lab frame,
        as well as their z-positions."""
        # Construct the batch of `AbstractPose`s
        cls = type(self.pose)
        make_poses = jax.vmap(
            lambda rot, pos: cls.from_rotation_and_translation(rot, pos)
        )
        positions_in_body_frame = self.positions_in_body_frame
        poses = make_poses(self.rotations_in_body_frame, positions_in_body_frame[:, :2])
        z_positions = positions_in_body_frame[:, 2]
        return poses, z_positions

    def get_poses_and_z_positions_in_lab_frame(
        self,
    ) -> tuple[AbstractPose, Float[Array, " {self.n_subcomponents}"]]:
        """Draw the poses of the subcomponents in the lab frame,
        as well as their z-positions."""
        # Construct the batch of `AbstractPose`s
        cls = type(self.pose)
        make_poses = jax.vmap(
            lambda rot, pos: cls.from_rotation_and_translation(rot, pos)
        )
        positions_in_lab_frame = self.positions_in_lab_frame
        poses = make_poses(self.rotations_in_lab_frame, positions_in_lab_frame[:, :2])
        z_positions = positions_in_lab_frame[:, 2]
        return poses, z_positions


class AbstractAssemblyWithSubunit(AbstractAssembly, strict=True):
    """Abstraction of a biological assembly with a single
    assymmetric subunit (ASU).
    """

    subunit: AbstractVar[AbstractStructuralEnsemble]

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
    def get_subcomponents_and_z_positions_in_body_frame(
        self,
    ) -> tuple[AbstractStructuralEnsemble, Float[Array, " {self.n_subcomponents}"]]:
        """Draw a realization of all of the subunits in the body frame."""
        poses, z_postions = self.get_poses_and_z_positions_in_body_frame()
        subunits = _make_subunits(self.subunit, self.conformation, poses)
        return subunits, z_postions

    @override
    def get_subcomponents_and_z_positions_in_lab_frame(
        self,
    ) -> tuple[AbstractStructuralEnsemble, Float[Array, " {self.n_subcomponents}"]]:
        """Draw a realization of all of the subunits in the lab frame."""
        poses, z_postions = self.get_poses_and_z_positions_in_lab_frame()
        subunits = _make_subunits(self.subunit, self.conformation, poses)
        return subunits, z_postions

    @property
    def n_subunits(self) -> int:
        return self.n_subcomponents

    @cached_property
    def subunits_in_body_frame(self) -> AbstractStructuralEnsemble:
        """Convenience method for grabbing the subunits from the method
        `AbstractAssemblyWithSubunit.get_subcomponents_and_z_positions_in_body_frame`.
        """
        return self.get_subcomponents_and_z_positions_in_body_frame()[0]

    @cached_property
    def subunits_in_lab_frame(self) -> AbstractStructuralEnsemble:
        """Convenience method for grabbing the subunits from the method
        `AbstractAssemblyWithSubunit.get_subcomponents_and_z_positions_in_lab_frame`.
        """
        return self.get_subcomponents_and_z_positions_in_lab_frame()[0]


def _make_subunits(subunit, conformation, poses):
    if subunit.conformation is not None:
        where = lambda s: (s.conformation, s.pose)
        return eqx.tree_at(where, subunit, (conformation, poses))
    else:
        where = lambda s: s.pose
        return eqx.tree_at(where, subunit, poses)

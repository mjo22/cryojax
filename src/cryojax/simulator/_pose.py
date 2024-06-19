"""
Representations of rigid body rotations and translations of 3D coordinate systems.
"""

from abc import abstractmethod
from functools import cached_property
from typing import overload
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float

from ..rotations import convert_quaternion_to_euler_angles, SO3


class AbstractPose(Module, strict=True):
    """Base class for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1. Define angular coordinates as dataclass fields and
           write `__init__`.

        2. Overwrite the `AbstractPose.rotation` property and
           `AbstractPose.from_rotation` class method.
    """

    offset_x_in_angstroms: AbstractVar[Float[Array, ""]]
    offset_y_in_angstroms: AbstractVar[Float[Array, ""]]
    offset_z_in_angstroms: AbstractVar[Float[Array, ""]]

    @overload
    def rotate_coordinates(
        self,
        volume_coordinates: Float[Array, "z_dim y_dim x_dim 3"],
        inverse: bool = False,
    ) -> Float[Array, "z_dim y_dim x_dim 3"]: ...

    @overload
    def rotate_coordinates(
        self, volume_coordinates: Float[Array, "size 3"], inverse: bool = False
    ) -> Float[Array, "size 3"]: ...

    def rotate_coordinates(
        self,
        volume_coordinates: Float[Array, "z_dim y_dim x_dim 3"] | Float[Array, "size 3"],
        inverse: bool = False,
    ) -> Float[Array, "z_dim y_dim x_dim 3"] | Float[Array, "size 3"]:
        """Rotate coordinates from a particular convention."""
        rotation = self.rotation.inverse() if inverse else self.rotation
        if isinstance(volume_coordinates, Float[Array, "size 3"]):  # type: ignore
            rotated_volume_coordinates = jax.vmap(rotation.apply)(volume_coordinates)
        elif isinstance(volume_coordinates, Float[Array, "z_dim y_dim x_dim 3"]):  # type: ignore
            rotated_volume_coordinates = jax.vmap(jax.vmap(jax.vmap(rotation.apply)))(
                volume_coordinates
            )
        else:
            raise ValueError(
                "Coordinates must be a JAX array either of shape (N, 3) or "
                f"(N1, N2, N3, 3). Instead, got {volume_coordinates.shape} and type "
                f"{type(volume_coordinates)}."
            )
        return rotated_volume_coordinates

    def compute_shifts(
        self, frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Compute the phase shifts from the in-plane translation,
        given a frequency grid coordinate system.
        """
        xy = self.offset_in_angstroms[0:2]
        return jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid_in_angstroms, xy)))

    @cached_property
    def offset_in_angstroms(self) -> Float[Array, "3"]:
        """The translation vector, relative to the center of the in-plane
        (x, y) coordinates and relative to the configured defocus in the
        out-of-plane z coordinate.
        """
        return jnp.asarray(
            (
                self.offset_x_in_angstroms,
                self.offset_y_in_angstroms,
                self.offset_z_in_angstroms,
            )
        )

    @cached_property
    @abstractmethod
    def rotation(self) -> SO3:
        """Generate an `SO3` object."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_rotation(cls, rotation: SO3):
        """Construct an `AbstractPose` from an `SO3` object."""
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        offset_in_angstroms: Float[Array, "3"],
    ):
        """Construct an `AbstractPose` from an `AbstractRotation` object and a
        translation vector.
        """
        return eqx.tree_at(
            lambda self: (
                self.offset_x_in_angstroms,
                self.offset_y_in_angstroms,
                self.offset_z_in_angstroms,
            ),
            cls.from_rotation(rotation),
            (
                offset_in_angstroms[..., 0],
                offset_in_angstroms[..., 1],
                offset_in_angstroms[..., 2],
            ),
        )


class EulerAnglePose(AbstractPose, strict=True):
    r"""An `AbstractPose` represented by Euler angles.
    Angles are given in degrees, and the sequence of rotations is
    given by a zyz extrinsic rotations.
    """

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]
    offset_z_in_angstroms: Float[Array, ""]

    view_phi: Float[Array, ""]
    view_theta: Float[Array, ""]
    view_psi: Float[Array, ""]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_z_in_angstroms: float | Float[Array, ""] = 0.0,
        view_phi: float | Float[Array, ""] = 0.0,
        view_theta: float | Float[Array, ""] = 0.0,
        view_psi: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `offset_z_in_angstroms`:
            Out-of-plane translation in the z direction. The translation is measured
            relative to the configured defocus.
        - `view_phi`: Angle to rotate about first rotation axis, which is the z axis.
        - `view_theta`: Angle to rotate about second rotation axis, which is the y axis.
        - `view_psi`: Angle to rotate about third rotation axis, which is the z axis.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.offset_z_in_angstroms = jnp.asarray(offset_z_in_angstroms)
        self.view_phi = jnp.asarray(view_phi)
        self.view_theta = jnp.asarray(view_theta)
        self.view_psi = jnp.asarray(view_psi)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate a `SO3` object from a set of Euler angles."""
        phi, theta, psi = self.view_phi, self.view_theta, self.view_psi
        # Convert to radians.
        phi = jnp.deg2rad(phi)
        theta = jnp.deg2rad(theta)
        psi = jnp.deg2rad(psi)
        # Get sequence of rotations.
        R1, R2, R3 = (
            SO3.from_z_radians(phi),
            SO3.from_y_radians(theta),
            SO3.from_z_radians(psi),
        )
        return R3 @ R2 @ R1

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        view_phi, view_theta, view_psi = convert_quaternion_to_euler_angles(
            rotation.wxyz,
            convention="zyz",
        )
        return cls(view_phi=view_phi, view_theta=view_theta, view_psi=view_psi)


class QuaternionPose(AbstractPose, strict=True):
    """An `AbstractPose` represented by unit quaternions."""

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]
    offset_z_in_angstroms: Float[Array, ""]

    wxyz: Float[Array, "4"]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_z_in_angstroms: float | Float[Array, ""] = 0.0,
        wxyz: (
            tuple[float, float, float, float] | Float[np.ndarray, "4"] | Float[Array, "4"]
        ) = (1.0, 0.0, 0.0, 0.0),
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `offset_z_in_angstroms`:
            Out-of-plane translation in the z direction. The translation is measured
            relative to the configured defocus.
        - `wxyz`:
            The quaternion, represented as a vector $\\mathbf{q} = (q_w, q_x, q_y, q_z)$.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.offset_z_in_angstroms = jnp.asarray(offset_z_in_angstroms)
        self.wxyz = jnp.asarray(wxyz)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from the unit quaternion."""
        # Generate SO3 object from unit quaternion
        R = SO3(wxyz=self.wxyz).normalize()
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(wxyz=rotation.wxyz)


class AxisAnglePose(AbstractPose, strict=True):
    """An `AbstractPose` parameterized in the axis-angle representation.

    The axis-angle representation parameterizes elements of the so3 algebra,
    which are skew-symmetric matrices, with the euler vector. The magnitude
    of this vector is the angle, and the unit vector is the axis.

    In a `SO3` object, the euler vector is mapped to SO3 group elements using
    the matrix exponential.
    """

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]
    offset_z_in_angstroms: Float[Array, ""]

    euler_vector: Float[Array, "3"]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_z_in_angstroms: float | Float[Array, ""] = 0.0,
        euler_vector: (
            tuple[float, float, float] | Float[np.ndarray, "3"] | Float[Array, "3"]
        ) = (0.0, 0.0, 0.0),
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `offset_z_in_angstroms`:
            Out-of-plane translation in the z direction. The translation is measured
            relative to the configured defocus.
        - `euler_vector`:
            The axis-angle parameterization, represented as a
            vector $\\boldsymbol{\\omega} = (\\omega_x, \\omega_y, \\omega_z)$.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.offset_z_in_angstroms = jnp.asarray(offset_z_in_angstroms)
        self.euler_vector = jnp.asarray(euler_vector)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from an euler vector using the exponential map."""
        # Convert degrees to radians
        euler_vector = jnp.deg2rad(self.euler_vector)
        # Project the tangent vector onto the manifold with
        # the exponential map
        R = SO3.exp(euler_vector)
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        # Compute the euler vector from the logarithmic map
        euler_vector = jnp.rad2deg(rotation.log())
        return cls(euler_vector=euler_vector)

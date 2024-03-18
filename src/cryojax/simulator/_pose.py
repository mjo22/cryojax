"""
Representations of rigid body rotations and translations of 3D coordinate systems.
"""

from abc import abstractmethod
from typing import overload, Any
from typing_extensions import override
from jaxtyping import Float, Array
from functools import cached_property
from equinox import field, AbstractVar

import jax
import equinox as eqx
import jax.numpy as jnp
from equinox import Module

from ..rotations import SO3
from ..typing import (
    RealNumber,
    ComplexImage,
    ImageCoords,
    PointCloudCoords3D,
    VolumeCoords,
)


class AbstractPose(Module, strict=True):
    """Base class for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1. Define angular coordinates as dataclass fields, along
           with other dataclass fields.

        2. Overwrite the `AbstractPose.rotation` property and
           `AbstractPose.from_rotation` class method.

    **Attributes**:

    - `offset_x_in_angstroms` : In-plane translation in x direction.

    - `offset_y_in_angstroms` : In-plane translation in y direction.

    - `offset_z_in_angstroms` : Out-of-plane translation in the z
                                direction. The translation is measured
                                relative to the configured defocus.
    """

    offset_x_in_angstroms: AbstractVar[RealNumber]
    offset_y_in_angstroms: AbstractVar[RealNumber]
    offset_z_in_angstroms: AbstractVar[RealNumber]

    @overload
    def rotate_coordinates(
        self, volume_coordinates: VolumeCoords, inverse: bool = False
    ) -> VolumeCoords: ...

    @overload
    def rotate_coordinates(
        self, volume_coordinates: PointCloudCoords3D, inverse: bool = False
    ) -> PointCloudCoords3D: ...

    def rotate_coordinates(
        self,
        volume_coordinates: VolumeCoords | PointCloudCoords3D,
        inverse: bool = False,
    ) -> VolumeCoords | PointCloudCoords3D:
        """Rotate coordinates from a particular convention."""
        rotation = self.rotation.inverse() if inverse else self.rotation
        if isinstance(volume_coordinates, PointCloudCoords3D):
            rotated_volume_coordinates = jax.vmap(rotation.apply)(volume_coordinates)
        elif isinstance(volume_coordinates, VolumeCoords):
            rotated_volume_coordinates = jax.vmap(jax.vmap(jax.vmap(rotation.apply)))(
                volume_coordinates
            )
        else:
            raise ValueError(
                "Coordinates must be a JAX array either of shape (N, 3) or (N1, N2, N3, 3). "
                f"Instead, got {volume_coordinates.shape} and type {type(volume_coordinates)}."
            )
        return rotated_volume_coordinates

    def compute_shifts(self, frequency_grid_in_angstroms: ImageCoords) -> ComplexImage:
        """Compute the phase shifts from the in-plane translation,
        given a frequency grid coordinate system.
        """
        xy = self.offset_in_angstroms[0:2]
        return jnp.exp(
            -1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid_in_angstroms, xy))
        )

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
    def from_rotation(cls, rotation: SO3, **kwargs: Any):
        """Construct an `AbstractPose` from an `SO3` object."""
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        offset_in_angstroms: Float[Array, "3"],
        **kwargs: Any,
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
            cls.from_rotation(rotation, **kwargs),
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

    **Attributes:**

    - `view_phi`: Angle to rotate about first rotation axis, which is the z axis.
    - `view_theta`: Angle to rotate about second rotation axis, which is the y axis.
    - `view_psi`: Angle to rotate about third rotation axis, which is the z axis.
    """

    offset_x_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)

    view_phi: RealNumber = field(default=0.0, converter=jnp.asarray)
    view_theta: RealNumber = field(default=0.0, converter=jnp.asarray)
    view_psi: RealNumber = field(default=0.0, converter=jnp.asarray)

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
        view_phi, view_theta, view_psi = _convert_quaternion_to_euler_angles(
            rotation.wxyz,
            "zyz",
        )
        return cls(view_phi=view_phi, view_theta=view_theta, view_psi=view_psi)


class QuaternionPose(AbstractPose, strict=True):
    """
    An `AbstractPose` represented by unit quaternions.

    **Attributes:**

    - `wxyz`: The unit quaternion.
    """

    offset_x_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)

    wxyz: Float[Array, "4"] = field(default=(1.0, 0.0, 0.0, 0.0), converter=jnp.asarray)

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

    **Attributes:**

    - `euler_vector`: The axis-angle parameterization.
    """

    offset_x_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: RealNumber = field(default=0.0, converter=jnp.asarray)

    euler_vector: Float[Array, "3"] = field(
        default=(0.0, 0.0, 0.0), converter=jnp.asarray
    )

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


def _convert_quaternion_to_euler_angles(
    wxyz: jax.Array, convention: str = "zyz"
) -> jax.Array:
    """Convert a quaternion to a sequence of euler angles about an extrinsic
    coordinate system.

    Adapted from https://github.com/chrisflesher/jax-scipy-spatial/.
    """
    if len(convention) != 3 or not all(
        [axis in ["x", "y", "z"] for axis in convention]
    ):
        raise ValueError(
            f"`convention` should be a string of three characters, each "
            f"of which is 'x', 'y', or 'z'. Instead, got '{convention}'"
        )
    if convention[0] == convention[1] or convention[1] == convention[2]:
        raise ValueError(
            f"`convention` cannot have axes repeating in a row. For example, "
            f"'xxy' or 'zzz' are not allowed. Got '{convention}'."
        )
    xyz_axis_to_array_axis = {"x": 0, "y": 1, "z": 2}
    axes = [xyz_axis_to_array_axis[axis] for axis in convention]
    xyzw = jnp.roll(wxyz, shift=-1)
    angle_first = 0
    angle_third = 2
    i = axes[0]
    j = axes[1]
    k = axes[2]
    symmetric = i == k
    k = jnp.where(symmetric, 3 - i - j, k)
    sign = jnp.array((i - j) * (j - k) * (k - i) // 2, dtype=xyzw.dtype)
    eps = 1e-7
    a = jnp.where(symmetric, xyzw[3], xyzw[3] - xyzw[j])
    b = jnp.where(symmetric, xyzw[i], xyzw[i] + xyzw[k] * sign)
    c = jnp.where(symmetric, xyzw[j], xyzw[j] + xyzw[3])
    d = jnp.where(symmetric, xyzw[k] * sign, xyzw[k] * sign - xyzw[i])
    angles = jnp.empty(3, dtype=xyzw.dtype)
    angles = angles.at[1].set(2 * jnp.arctan2(jnp.hypot(c, d), jnp.hypot(a, b)))
    case = jnp.where(jnp.abs(angles[1] - jnp.pi) <= eps, 2, 0)
    case = jnp.where(jnp.abs(angles[1]) <= eps, 1, case)
    half_sum = jnp.arctan2(b, a)
    half_diff = jnp.arctan2(d, c)
    angles = angles.at[0].set(
        jnp.where(case == 1, 2 * half_sum, 2 * half_diff * -1)
    )  # any degenerate case
    angles = angles.at[angle_first].set(
        jnp.where(case == 0, half_sum - half_diff, angles[angle_first])
    )
    angles = angles.at[angle_third].set(
        jnp.where(case == 0, half_sum + half_diff, angles[angle_third])
    )
    angles = angles.at[angle_third].set(
        jnp.where(symmetric, angles[angle_third], angles[angle_third] * sign)
    )
    angles = angles.at[1].set(jnp.where(symmetric, angles[1], angles[1] - jnp.pi / 2))
    angles = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return -jnp.rad2deg(angles)

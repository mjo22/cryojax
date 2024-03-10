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

from ..rotations import SO3, AbstractRotation
from ..typing import (
    Real_,
    ComplexImage,
    ImageCoords,
    CloudCoords3D,
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

    offset_x_in_angstroms: AbstractVar[Real_]
    offset_y_in_angstroms: AbstractVar[Real_]
    offset_z_in_angstroms: AbstractVar[Real_]

    inverse: AbstractVar[bool]

    @overload
    def rotate_coordinates(
        self, volume_coordinates: VolumeCoords, inverse: bool = False
    ) -> VolumeCoords: ...

    @overload
    def rotate_coordinates(
        self, volume_coordinates: CloudCoords3D, inverse: bool = False
    ) -> CloudCoords3D: ...

    def rotate_coordinates(
        self,
        volume_coordinates: VolumeCoords | CloudCoords3D,
        inverse: bool = False,
    ) -> VolumeCoords | CloudCoords3D:
        """Rotate coordinates from a particular convention."""
        rotation = self.rotation.inverse() if inverse else self.rotation
        if isinstance(volume_coordinates, CloudCoords3D):
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
    def rotation(self) -> AbstractRotation:
        """Generate a `AbstractRotation` object."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_rotation(cls, rotation: AbstractRotation, **kwargs: Any):
        """Construct an `AbstractPose` from a `AbstractRotation` object."""
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: AbstractRotation,
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

    **Attributes:**

    - `view_phi`:
        Angle to rotate about first rotation axis.

    - `view_theta`:
        Angle to rotate about second rotation axis.

    - `view_psi`:
        Angle to rotate about third rotation axis.

    - `convention`:
        The sequence of axes over which to apply
        rotation. This is a string of 3 characters
        of x, y, and z. By default, `'zyz'`.

    - `inverse`:
        Compute the inverse rotation of the specified
        convention. By default, ``False``. The value
        of this argument is with respect to fourier space
        rotations, so it is automatically inverted
        when rotating in real space.
    """

    offset_x_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)

    view_phi: Real_ = field(default=0.0, converter=jnp.asarray)
    view_theta: Real_ = field(default=0.0, converter=jnp.asarray)
    view_psi: Real_ = field(default=0.0, converter=jnp.asarray)

    inverse: bool = field(static=True, default=False)
    convention: str = field(static=True, default="zyz")
    degrees: bool = field(static=True, default=True)

    def __check_init__(self):
        if len(self.convention) != 3 or not all(
            [axis in ["x", "y", "z"] for axis in self.convention]
        ):
            raise AttributeError(
                f"`EulerPose.convention` should be a string of three characters, each "
                f"of which is 'x', 'y', or 'z'. Instead, got '{self.convention}'"
            )
        if (
            self.convention[0] == self.convention[1]
            or self.convention[1] == self.convention[2]
        ):
            raise AttributeError(
                f"`EulerPose.convention` cannot have axes repeating in a row. For example, "
                f"'xxy' or 'zzz' are not allowed. Got '{self.convention}'."
            )

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate a `SO3` object from a set of Euler angles."""
        phi, theta, psi = self.view_phi, self.view_theta, self.view_psi
        # Convert to radians.
        if self.degrees:
            phi = jnp.deg2rad(phi)
            theta = jnp.deg2rad(theta)
            psi = jnp.deg2rad(psi)
        # Get sequence of rotations.
        R1, R2, R3 = [
            getattr(SO3, f"from_{axis}_radians")(angle)
            for axis, angle in zip(self.convention, [phi, theta, psi])
        ]
        R = R3 @ R2 @ R1
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(
        cls, rotation: SO3, convention: str = "zyz", degrees: bool = True
    ):
        view_phi, view_theta, view_psi = _convert_quaternion_to_euler_angles(
            rotation.wxyz, convention, degrees
        )
        return cls(
            view_phi=view_phi,
            view_theta=view_theta,
            view_psi=view_psi,
            convention=convention,
            degrees=degrees,
        )


class QuaternionPose(AbstractPose, strict=True):
    """
    An `AbstractPose` represented by unit quaternions.

    **Attributes:**

    - `wxyz`: The unit quaternion.
    """

    offset_x_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)

    wxyz: Float[Array, "4"] = field(default=(1.0, 0.0, 0.0, 0.0), converter=jnp.asarray)

    inverse: bool = field(static=True, default=False)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from the unit quaternion."""
        # Generate SO3 object from unit quaternion
        R = SO3(wxyz=(self.wxyz / jnp.linalg.norm(self.wxyz)))
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(wxyz=rotation.wxyz)


class MatrixPose(AbstractPose, strict=True):
    """An `AbstractPose` represented by a rotation matrix.

    **Attributes:**

    - `matrix`: The rotation matrix.
    """

    offset_x_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)

    rotation_matrix: Float[Array, "3 3"] = field(
        default_factory=lambda: jnp.eye(3), converter=jnp.asarray
    )

    inverse: bool = field(static=True, default=False)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from a rotation matrix."""
        # Generate SO3 object from the rotation matrix
        R = SO3.from_matrix(self.rotation_matrix)
        # Normalize (this should be equivalent to making sure the
        # rotation matrix has unit determinant)
        R = R.normalize()
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(rotation_matrix=rotation.as_matrix())


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

    offset_x_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)

    euler_vector: Float[Array, "3"] = field(
        default=(0.0, 0.0, 0.0), converter=jnp.asarray
    )

    inverse: bool = field(static=True, default=False)
    degrees: bool = field(static=True, default=True)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from an euler vector using the exponential map."""
        euler_vector = self.euler_vector
        if self.degrees:
            euler_vector = jnp.deg2rad(euler_vector)
        # Project the tangent vector onto the manifold with
        # the exponential map
        R = SO3.exp(euler_vector)
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3, degrees: bool = True):
        # Compute the euler vector from the logarithmic map
        euler_vector = rotation.log()
        if degrees:
            euler_vector = jnp.rad2deg(euler_vector)
        return cls(euler_vector=euler_vector)


class SO3Pose(AbstractPose, strict=True):
    """An `AbstractPose` represented by a `SO3` matrix lie
    group.

    **Attributes:**

    `group_element`: The group element of SO3.
    """

    offset_x_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z_in_angstroms: Real_ = field(default=0.0, converter=jnp.asarray)

    group_element: SO3 = field(
        default_factory=lambda: SO3(wxyz=jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    )

    inverse: bool = field(static=True, default=False)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Get the `jaxlie.SO3` matrix lie group."""
        R = self.group_element
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(group_element=rotation)


def _convert_quaternion_to_euler_angles(
    wxyz: jax.Array, convention: str, degrees: bool
) -> jax.Array:
    """Convert a quaternion to a sequence of euler angles.

    Adapted from https://github.com/chrisflesher/jax-scipy-spatial/.
    """
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
    return -jnp.rad2deg(angles) if degrees else -angles

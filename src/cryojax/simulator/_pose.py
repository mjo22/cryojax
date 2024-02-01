"""
Routines that compute coordinate rotations and translations.
"""

from abc import abstractmethod
from typing import Union, overload
from typing_extensions import override
from jaxtyping import Float, Array
from functools import cached_property
from equinox import field, AbstractVar

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxlie import SO3
from equinox import Module

from ..typing import (
    Real_,
    ComplexImage,
    ImageCoords,
    CloudCoords3D,
    VolumeCoords,
)


class AbstractPose(Module, strict=True):
    """
    Base class for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1) Define angular coordinates.
        2) Overwrite the ``AbstractPose.rotation`` property.

    Attributes
    ----------`
    offset_x :
        In-plane translation in x direction.
    offset_y :
        In-plane translation in y direction.
    offset_z :
        Out-of-plane translation in the z
        direction. The translation is measured
        relative to the configured defocus.
    """

    offset_x: AbstractVar[Real_]
    offset_y: AbstractVar[Real_]
    offset_z: AbstractVar[Real_]

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
        """
        Rotate coordinates from a particular convention.

        By default, compute the inverse rotation if rotating in
        real-space.
        """
        rotation = self.rotation.inverse() if inverse else self.rotation
        shape = volume_coordinates.shape
        if isinstance(volume_coordinates, CloudCoords3D):
            rotated_coordinates = jax.vmap(rotation.apply)(volume_coordinates)
        elif isinstance(volume_coordinates, VolumeCoords):
            N1, N2, N3 = shape[:-1]
            rotated_coordinates = jax.vmap(rotation.apply)(
                volume_coordinates.reshape(N1 * N2 * N3, 3)
            )
            rotated_coordinates = rotated_coordinates.reshape((N1, N2, N3, 3))
        else:
            raise ValueError(
                f"Coordinates must be a JAX array either of shape (N, 3) or (N1, N2, N3, 3). Instead, got {volume_coordinates.shape} and type {type(volume_coordinates)}."
            )
        return rotated_coordinates

    def compute_shifts(self, frequency_grid: ImageCoords) -> ComplexImage:
        """
        Compute the phase shifts from the in-plane translation,
        given a frequency grid coordinate system.
        """
        xy = self.offset[0:2]
        return jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, xy)))

    @cached_property
    def offset(self) -> Float[Array, "3"]:
        """The translation vector."""
        return jnp.asarray((self.offset_x, self.offset_y, self.offset_z))

    @cached_property
    @abstractmethod
    def rotation(self) -> SO3:
        """Generate a rotation."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_rotation(cls, rotation: SO3):
        """
        Construct a ``Pose`` from a ``jaxlie.SO3`` object.
        """
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls, rotation: SO3, translation: Float[Array, "3"]
    ):
        """
        Construct a ``Pose`` from a ``jaxlie.SO3`` object and a
        translation vector.
        """
        return eqx.tree_at(
            lambda self: (self.offset_x, self.offset_y, self.offset_z),
            cls.from_rotation(rotation),
            (translation[..., 0], translation[..., 1], translation[..., 2]),
        )


class EulerPose(AbstractPose, strict=True):
    r"""
    An image pose using Euler angles.

    Attributes
    ----------
    convention :
        The sequence of axes over which to apply
        rotation. This is a string of 3 characters
        of x, y, and z. By default, `zyz`.
    intrinsic :
        If ``True``, follow the intrinsic rotation
        convention. If ``False``, rotation axes move with
        each rotation.
    inverse :
        Compute the inverse rotation of the specified
        convention. By default, ``False``. The value
        of this argument is with respect to fourier space
        rotations, so it is automatically inverted
        when rotating in real space.
    view_phi :
        First rotation axis, ranging :math:`(-\pi, \pi]`.
    view_theta :
        Second rotation axis, ranging :math:`(-\pi, \pi]`.
    view_psi :
        Third rotation axis, ranging :math:`(-\pi, \pi]`.
    """

    offset_x: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z: Real_ = field(default=0.0, converter=jnp.asarray)

    view_phi: Real_ = field(default=0.0, converter=jnp.asarray)
    view_theta: Real_ = field(default=0.0, converter=jnp.asarray)
    view_psi: Real_ = field(default=0.0, converter=jnp.asarray)

    inverse: bool = field(static=True, default=False)
    convention: str = field(static=True, default="zyz")
    degrees: bool = field(static=True, default=True)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate a rotation from a set of Euler angles."""
        R = make_euler_rotation(
            self.view_phi,
            self.view_theta,
            self.view_psi,
            degrees=self.degrees,
            convention=self.convention,
        )
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        raise NotImplementedError(
            "Cannot convert SO3 object to arbitrary Euler angle convention. See https://github.com/brentyi/jaxlie/issues/16"
        )


class QuaternionPose(AbstractPose, strict=True):
    """
    An image pose using unit Quaternions.

    Attributes
    ----------
    wxyz :
    """

    offset_x: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z: Real_ = field(default=0.0, converter=jnp.asarray)

    wxyz: Float[Array, "4"] = field(
        default=(1.0, 0.0, 0.0, 0.0), converter=jnp.asarray
    )

    inverse: bool = field(static=True, default=False)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from a unit quaternion."""
        q_norm = jnp.linalg.norm(self.wxyz)
        R = SO3(wxyz=self.wxyz / q_norm)
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(wxyz=rotation.wxyz)


class MatrixPose(AbstractPose, strict=True):
    """
    An image pose represented by a rotation matrix.

    Attributes
    ----------
    matrix :
        The rotation matrix.
    """

    offset_x: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_y: Real_ = field(default=0.0, converter=jnp.asarray)
    offset_z: Real_ = field(default=0.0, converter=jnp.asarray)

    matrix: Float[Array, "3 3"] = field(
        default_factory=lambda: jnp.eye(3), converter=jnp.asarray
    )

    inverse: bool = field(static=True, default=False)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from a rotation matrix."""
        R = SO3.from_matrix(self.matrix)
        return R.inverse() if self.inverse else R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3):
        return cls(matrix=rotation.as_matrix())


def make_euler_rotation(
    phi: Union[float, Real_],
    theta: Union[float, Real_],
    psi: Union[float, Real_],
    convention: str = "zyz",
    degrees: bool = False,
) -> SO3:
    """Helper routine to generate a rotation in a particular
    convention.
    """
    # Convert to radians.
    if degrees:
        phi = jnp.deg2rad(phi)
        theta = jnp.deg2rad(theta)
        psi = jnp.deg2rad(psi)
    # Get sequence of rotations, converting to cryojax conventions
    # from jaxlie
    R1, R2, R3 = [
        getattr(SO3, f"from_{axis}_radians")(angle).inverse()
        for axis, angle in zip(convention, [phi, theta, psi])
    ]
    return R3 @ R2 @ R1

"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = [
    "rotate_coordinates",
    "compute_shifts",
    "make_euler_rotation",
    "Pose",
    "EulerPose",
    "QuaternionPose",
]

from abc import abstractmethod
from typing import Union, Optional, Any
from typing_extensions import override
from jaxtyping import Float, Array
from functools import cached_property

import jax
import jax.numpy as jnp
from jaxlie import SO3

from ..core import field, Module
from ..typing import (
    Real_,
    ComplexImage,
    ImageCoords,
    CloudCoords3D,
    VolumeCoords,
)

_RotationMatrix3D = Float[Array, "3 3"]
_Vector3D = Float[Array, "3"]
_Vector2D = Float[Array, "2"]


class Pose(Module):
    """
    Base class for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1) Define angular coordinates.
        2) Overwrite the ``Pose.rotation`` property.

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

    offset_x: Real_ = field(default=0.0)
    offset_y: Real_ = field(default=0.0)
    offset_z: Real_ = field(default=0.0)

    def rotate(
        self,
        coordinates: Union[VolumeCoords, CloudCoords3D],
        is_real: bool = True,
    ) -> Union[VolumeCoords, CloudCoords3D]:
        """
        Rotate coordinates from a particular convention.

        By default, compute the inverse rotation if rotating in
        real-space.
        """
        rotation = self.rotation.inverse() if is_real else self.rotation
        return rotate_coordinates(coordinates, rotation)

    def shifts(self, freqs: ImageCoords) -> ComplexImage:
        """
        Compute the phase shifts from the in-plane translation,
        given a wave vector coordinate system.
        """
        xy = self.offset[0:2]
        return compute_shifts(freqs, xy)

    @cached_property
    def offset(self) -> _Vector3D:
        """The translation vector."""
        return jnp.asarray((self.offset_x, self.offset_y, self.offset_z))

    @cached_property
    @abstractmethod
    def rotation(self) -> SO3:
        """Generate a rotation."""
        raise NotImplementedError


class EulerPose(Pose):
    """
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

    convention: str = field(static=True, default="zyz")
    intrinsic: bool = field(static=True, default=True)
    inverse: bool = field(static=True, default=False)
    degrees: bool = field(static=True, default=True)

    view_phi: Real_ = field(default=0.0)
    view_theta: Real_ = field(default=0.0)
    view_psi: Real_ = field(default=0.0)

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
            intrinsic=self.intrinsic,
        )
        return R.inverse() if self.inverse else R


class QuaternionPose(Pose):
    """
    An image pose using unit Quaternions.

    Attributes
    ----------
    view_qw :
    view_qx :
    view_qy :
    view_qz :
    """

    inverse: bool = field(static=True, default=False)

    view_qw: Real_ = field(default=1.0)
    view_qx: Real_ = field(default=0.0)
    view_qy: Real_ = field(default=0.0)
    view_qz: Real_ = field(default=0.0)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from a unit quaternion."""
        q = jnp.asarray(
            [self.view_qw, self.view_qx, self.view_qy, self.view_qz]
        )
        q_norm = jnp.sqrt(self.view_qx**2 + self.view_qx**2 + self.view_qy**2 + self.view_qz**2)
        R = SO3(wxyz=q / q_norm)
        return R.inverse() if self.inverse else R


class MatrixPose(Pose):
    """
    An image pose represented by a rotation matrix.

    Attributes
    ----------
    matrix :
        The rotation matrix.
    """

    matrix: _RotationMatrix3D = field()

    def __init__(
        self,
        *args: Any,
        matrix: Optional[_RotationMatrix3D] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.matrix = jnp.eye(3) if matrix is None else matrix

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from a rotation matrix."""
        return SO3.from_matrix(self.matrix)


def rotate_coordinates(
    coords: Union[VolumeCoords, CloudCoords3D],
    rotation: SO3,
) -> Union[VolumeCoords, CloudCoords3D]:
    r"""
    Compute a coordinate rotation.

    Arguments
    ---------
    coords :
        Coordinate system.
    rotation :
        The rotation object.

    Returns
    -------
    transformed :
        Rotated coordinate system.
    """
    shape = coords.shape
    if len(shape) == 2:
        transformed = jax.vmap(rotation.apply)(coords)
    elif len(shape) == 4:
        N1, N2, N3 = shape[0:-1]
        transformed = jax.vmap(rotation.apply)(coords.reshape(N1 * N2 * N3, 3))
        transformed = transformed.reshape((N1, N2, N3, 3))
    else:
        raise ValueError(
            "coords must either be shape (N, 3) or (N1, N2, N3, 3)"
        )

    return transformed


def compute_shifts(coords: ImageCoords, xy: _Vector2D) -> ComplexImage:
    r"""
    Compute the phase shifted density field from
    an in-plane real space translation.

    Arguments
    ---------
    coords :
        Coordinate system.
    xy :
        In-plane translation.

    Returns
    -------
    shifts :
        The phase shifts
    """
    return jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(coords, xy)))


def make_euler_rotation(
    phi: Union[float, Real_],
    theta: Union[float, Real_],
    psi: Union[float, Real_],
    convention: str = "zyz",
    intrinsic: bool = True,
    degrees: bool = False,
) -> SO3:
    """
    Helper routine to generate a rotation in a particular
    convention.
    """
    # Generate sequence of rotations
    rotations = [getattr(SO3, f"from_{axis}_radians") for axis in convention]
    if degrees:
        phi = jnp.deg2rad(phi)
        theta = jnp.deg2rad(theta)
        psi = jnp.deg2rad(psi)
    R1 = rotations[0](phi)
    R2 = rotations[1](theta)
    R3 = rotations[2](psi)
    R = R1 @ R2 @ R3 if intrinsic else R3 @ R2 @ R1

    return R

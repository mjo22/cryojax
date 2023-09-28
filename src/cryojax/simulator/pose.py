"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = [
    "rotate_rpy",
    "rotate_wxyz",
    "shift_phase",
    "make_euler_rotation",
    "Pose",
    "EulerPose",
    "QuaternionPose",
]

from abc import ABCMeta, abstractmethod
from typing import Any, Union

import jax
import jax.numpy as jnp
from jaxlie import SO3

from ..core import field, Module
from ..types import Real_, ComplexImage, ImageCoords, CloudCoords, VolumeCoords


class Pose(Module, metaclass=ABCMeta):
    """
    Base class PyTree container for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1) Define angular coordinates
        2) Overwrite the ``Pose.transform`` method.
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

    @abstractmethod
    def rotate(
        self,
        coordinates: Union[VolumeCoords, CloudCoords],
        real: bool = True,
    ) -> Union[VolumeCoords, CloudCoords]:
        """Rotation method for a particular pose convention."""
        raise NotImplementedError

    def shift(
        self, density: ComplexImage, coordinates: ImageCoords
    ) -> ComplexImage:
        """
        Translate a 2D electron density in real space by
        applying phase shifts in fourier space.
        """
        tx, ty = self.offset_x, self.offset_y
        shifted_density = shift_phase(
            density,
            coordinates,
            tx,
            ty,
        )
        return shifted_density


class EulerPose(Pose):
    """
    An image pose using Euler angles.

    Attributes
    ----------
    convention :
        The sequence of axes over which to apply
        rotation. This is a string of 3 characters
        of x, y, and z. By default, `zyx`.
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
        Roll angle, ranging :math:`(-\pi, \pi]`.
    view_theta :
        Pitch angle, ranging :math:`(-\pi, \pi]`.
    view_psi :
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    """

    convention: str = field(static=True, default="zyz")
    intrinsic: bool = field(static=True, default=True)
    inverse: bool = field(static=True, default=False)
    degrees: bool = field(static=True, default=True)

    view_phi: Real_ = field(default=0.0)
    view_theta: Real_ = field(default=0.0)
    view_psi: Real_ = field(default=0.0)

    def rotate(
        self,
        coordinates: Union[VolumeCoords, CloudCoords],
        real: bool = True,
    ) -> Union[VolumeCoords, CloudCoords]:
        """Rotate coordinates from a set of Euler angles."""
        if real:
            rotated, _ = rotate_rpy(
                coordinates,
                phi=self.view_phi,
                theta=self.view_theta,
                psi=self.view_psi,
                convention=self.convention,
                intrinsic=self.intrinsic,
                inverse=not self.inverse,
                degrees=self.degrees,
            )
            return rotated
        else:
            rotated, _ = rotate_rpy(
                coordinates,
                phi=self.view_phi,
                theta=self.view_theta,
                psi=self.view_psi,
                convention=self.convention,
                intrinsic=self.intrinsic,
                inverse=self.inverse,
                degrees=self.degrees,
            )
            return rotated


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

    def rotate(
        self,
        coordinates: Union[VolumeCoords, CloudCoords],
        real: bool = True,
    ) -> Union[VolumeCoords, CloudCoords]:
        """Rotate coordinates from a unit quaternion."""
        if real:
            rotated, _ = rotate_wxyz(
                coordinates,
                qw=self.view_qw,
                qx=self.view_qx,
                qy=self.view_qy,
                qz=self.view_qz,
                inverse=not self.inverse,
            )
            return rotated
        else:
            rotated, _ = rotate_wxyz(
                coordinates,
                qw=self.view_qw,
                qx=self.view_qx,
                qy=self.view_qy,
                qz=self.view_qz,
                inverse=self.inverse,
            )
            return rotated


def rotate_rpy(
    coords: Union[VolumeCoords, CloudCoords],
    phi: Real_,
    theta: Real_,
    psi: Real_,
    **kwargs: Any,
) -> tuple[Union[VolumeCoords, CloudCoords], SO3]:
    r"""
    Compute a coordinate rotation from
    a set of euler angles.

    Arguments
    ---------
    coords :
        Coordinate system.
    phi :
        First rotation axis, ranging :math:`(-\pi, \pi]`.
    theta :
        Second rotation axis, ranging :math:`(-\pi, \pi]`.
    psi :
        Third rotation axis, ranging :math:`(-\pi, \pi]`.
    kwargs :
        Keyword arguments passed to ``make_rpy_rotation``

    Returns
    -------
    transformed :
        Rotated and translated coordinate system.
    rotation : `jaxlie.SO3`
        The rotation.
    """
    coords = jnp.asarray(coords)
    shape = coords.shape
    rotation = make_euler_rotation(phi, theta, psi, **kwargs)
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

    return transformed, rotation


def rotate_wxyz(
    coords: Union[VolumeCoords, CloudCoords],
    qw: Real_,
    qx: Real_,
    qy: Real_,
    qz: Real_,
    inverse: bool = False,
) -> tuple[Union[VolumeCoords, CloudCoords], SO3]:
    r"""
    Compute a coordinate rotation from a quaternion.

    Arguments
    ---------
    coords :
        Coordinate system.
    qw :
    qx :
    qy :
    qz :

    Returns
    -------
    transformed :
        Rotated and translated coordinate system.
    rotation :
        The rotation.
    """
    coords = jnp.asarray(coords)
    shape = coords.shape
    wxyz = jnp.array([qw, qx, qy, qz])
    rotation = SO3.from_quaternion_xyzw(wxyz)
    rotation = rotation.inverse if inverse else rotation
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

    return transformed, rotation


def shift_phase(
    density: ComplexImage,
    coords: ImageCoords,
    tx: Real_,
    ty: Real_,
) -> ComplexImage:
    r"""
    Compute the phase shifted density field from
    an in-plane real space translation.

    Arguments
    ---------
    density :
        In-plane electron density in fourier
        space.
    coords :
        Coordinate system.
    tx :
        In-plane translation in x direction.
    ty :
        In-plane translation in y direction.

    Returns
    -------
    shifted : `Array`, shape `(N1, N2)`
        Shifted electron density.
    """
    coords, density = jnp.asarray(coords), jnp.asarray(density)
    xy = jnp.array([tx, ty])
    shift = jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(coords, xy)))
    shifted = density * shift

    return shifted


def make_euler_rotation(
    phi: Union[float, Real_],
    theta: Union[float, Real_],
    psi: Union[float, Real_],
    convention: str = "zyz",
    intrinsic: bool = True,
    inverse: bool = False,
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

    return R.inverse() if inverse else R

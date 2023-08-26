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
from typing import Any

import jax
import jax.numpy as jnp
from jaxlie import SO3

from ..core import Array, ArrayLike, Parameter, field, dataclass, CryojaxObject


@dataclass
class Pose(CryojaxObject, metaclass=ABCMeta):
    """
    Base class PyTree container for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1) Define angular coordinates
        2) Overwrite the ``Pose.transform`` method.
        3) Use the ``cryojax.core.dataclass`` decorator.
    Attributes
    ----------`
    offset_x : `cryojax.core.Parameter`
        In-plane translations in x direction.
    offset_y : `cryojax.core.Parameter`
        In-plane translations in y direction.
    """

    offset_x: Parameter = 0.0
    offset_y: Parameter = 0.0

    @abstractmethod
    def rotate(self, coordinates: ArrayLike, real: bool = True) -> Array:
        """Rotation method for a particular pose convention."""
        raise NotImplementedError

    def translate(self, density: ArrayLike, coordinates: ArrayLike) -> Array:
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


@dataclass
class EulerPose(Pose):
    """
    An image pose using Euler angles.

    Attributes
    ----------
    convention : `str`
        The sequence of axes over which to apply
        rotation. This is a string of 3 characters
        of x, y, and z. By default, `zyx`.
    intrinsic : `bool`
        If ``True``, follow the intrinsic rotation
        convention. If ``False``, rotation axes move with
        each rotation.
    inverse : `bool`
        Compute the inverse rotation of the specified
        convention. By default, ``False``. The value
        of this argument is with respect to fourier space
        rotations, so it is automatically inverted
        when rotating in real space.
    view_phi : `cryojax.core.Parameter`
        Roll angle, ranging :math:`(-\pi, \pi]`.
    view_theta : `cryojax.core.Parameter`
        Pitch angle, ranging :math:`(-\pi, \pi]`.
    view_psi : `cryojax.core.Parameter`
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    """

    convention: str = field(pytree_node=False, default="zyz")
    intrinsic: bool = field(pytree_node=False, default=True)
    inverse: bool = field(pytree_node=False, default=False)
    degrees: bool = field(pytree_node=False, default=True)

    view_phi: Parameter = 0.0
    view_theta: Parameter = 0.0
    view_psi: Parameter = 0.0

    def rotate(self, coordinates: ArrayLike, real: bool = True) -> Array:
        """Rotate coordinates from a set of Euler angles."""
        if real:
            rotated, _ = rotate_rpy(
                coordinates,
                *self.iter_data()[2:],
                convention=self.convention,
                intrinsic=self.intrinsic,
                inverse=not self.inverse,
                degrees=self.degrees,
            )
            return rotated
        else:
            rotated, _ = rotate_rpy(
                coordinates,
                *self.iter_data()[2:],
                convention=self.convention,
                intrinsic=self.intrinsic,
                inverse=self.inverse,
                degrees=self.degrees,
            )
            return rotated


@dataclass
class QuaternionPose(Pose):
    """
    An image pose using unit Quaternions.

    Attributes
    ----------
    view_qw : `cryojax.core.Parameter`
    view_qx : `cryojax.core.Parameter`
    view_qy : `cryojax.core.Parameter`
    view_qz : `cryojax.core.Parameter`
    """

    inverse: bool = field(pytree_node=False, default=False)

    view_qw: Parameter = 1.0
    view_qx: Parameter = 0.0
    view_qy: Parameter = 0.0
    view_qz: Parameter = 0.0

    def rotate(self, coordinates: ArrayLike, real: bool = True) -> Array:
        """Rotate coordinates from a unit quaternion."""
        if real:
            rotated, _ = rotate_wxyz(
                coordinates,
                *self.iter_data()[2:],
                inverse=not self.inverse,
            )
            return rotated
        else:
            rotated, _ = rotate_wxyz(
                coordinates,
                *self.iter_data()[2:],
                inverse=self.inverse,
            )
            return rotated


def rotate_rpy(
    coords: ArrayLike,
    phi: float,
    theta: float,
    psi: float,
    **kwargs: Any,
) -> tuple[Array, SO3]:
    r"""
    Compute a coordinate rotation from
    a set of euler angles.

    Arguments
    ---------
    coords : `Array`, shape `(N, 3)` or `(N1, N2, N3, 3)`
        Coordinate system.
    phi : `float`
        First rotation axis, ranging :math:`(-\pi, \pi]`.
    theta : `float`
        Second rotation axis, ranging :math:`(-\pi, \pi]`.
    psi : `float`
        Third rotation axis, ranging :math:`(-\pi, \pi]`.
    kwargs :
        Keyword arguments passed to ``make_rpy_rotation``

    Returns
    -------
    transformed : `Array`, shape `(N, 3)` or `(N1, N2, N3, 3)`
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
    coords: ArrayLike,
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    inverse: bool = False,
) -> tuple[Array, SO3]:
    r"""
    Compute a coordinate rotation from a quaternion.

    Arguments
    ---------
    coords : `Array` shape `(N, 3)` or `(N1, N2, N3, 3)`
        Coordinate system.
    qw : `float`
    qx : `float`
    qy : `float`
    qz : `float`

    Returns
    -------
    transformed : `Array`, shape `(N, 3)` or `(N1, N2, N3, 3)`
        Rotated and translated coordinate system.
    rotation : `jaxlie.SO3`
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
    density: ArrayLike,
    coords: ArrayLike,
    tx: float,
    ty: float,
) -> Array:
    r"""
    Compute the phase shifted density field from
    an in-plane real space translation.

    Arguments
    ---------
    density : `Array` shape `(N1, N2)`
        In-plane electron density in fourier
        space.
    coords : `Array` shape `(N1, N2, 2)`
        Coordinate system.
    tx : `float`
        In-plane translation in x direction.
    ty : `float`
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
    phi: float,
    theta: float,
    psi: float,
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

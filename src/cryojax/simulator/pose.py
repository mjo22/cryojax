"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = [
    "rotate_and_translate_rpy",
    "rotate_and_translate_wxyz",
    "rotate_rpy",
    "rotate_wxyz",
    "shift_phase",
    "Pose",
    "EulerPose",
    "QuaternionPose",
]

from abc import ABCMeta, abstractmethod
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3

from ..core import Array, Scalar, field, dataclass, CryojaxObject


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
    offset_x : `cryojax.core.Scalar`
        In-plane translations in x direction.
    offset_y : `cryojax.core.Scalar`
        In-plane translations in y direction.
    """

    offset_x: Scalar = 0.0
    offset_y: Scalar = 0.0

    @abstractmethod
    def transform(
        density: Array, coordinates: Array, real: bool = True
    ) -> Array:
        """Transformation method for a particular pose convention."""
        raise NotImplementedError


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
    fixed : `bool`
        If ``False``, axes rotation axes move with
        each rotation.
    view_phi : `cryojax.core.Scalar`
        Roll angles, ranging :math:`(-\pi, \pi]`.
    view_theta : `cryojax.core.Scalar`
        Pitch angles, ranging :math:`(0, \pi]`.
    view_psi : `cryojax.core.Scalar`
        Yaw angles, ranging :math:`(-\pi, \pi]`.
    """

    convention: str = field(pytree_node=False, default="zyx")
    fixed: bool = field(pytree_node=False, default=False)

    view_phi: Scalar = 0.0
    view_theta: Scalar = jnp.pi / 2
    view_psi: Scalar = 0.0

    def transform(
        self, density: Array, coordinates: Array, real: bool = True
    ) -> Array:
        """Transform coordinates from a set of Euler angles."""
        if real:
            return density, rotate_and_translate_rpy(
                coordinates,
                *self.iter_data(),
                convention=self.convention,
                fixed=self.fixed,
            )
        else:
            rotated_coordinates = rotate_rpy(
                coordinates,
                *self.iter_data()[2:],
                convention=self.convention,
                fixed=self.fixed,
            )
            shifted_density = shift_phase(
                density,
                coordinates,
                *self.iter_data()[:2],
            )
            return shifted_density, rotated_coordinates


@dataclass
class QuaternionPose(Pose):
    """
    An image pose using unit Quaternions.

    Attributes
    ----------
    view_qw : `cryojax.core.Scalar`
    view_qx : `cryojax.core.Scalar`
    view_qy : `cryojax.core.Scalar`
    view_qz : `cryojax.core.Scalar`
    """

    view_qw: Scalar = 1.0
    view_qx: Scalar = 0.0
    view_qy: Scalar = 0.0
    view_qz: Scalar = 0.0

    def transform(
        self, density: Array, coordinates: Array, real: bool = True
    ) -> Array:
        """Transform coordinates from an offset and unit quaternion."""
        if real:
            return density, rotate_and_translate_wxyz(
                coordinates, *self.iter_data()
            )
        else:
            rotated_coordinates = rotate_wxyz(
                coordinates, *self.iter_data()[2:]
            )
            shifted_density = shift_phase(
                density, coordinates, *self.iter_data()[:2]
            )
            return shifted_density, rotated_coordinates


@partial(jax.jit, static_argnames=["convention", "fixed"])
def rotate_and_translate_rpy(
    coords: Array,
    tx: float,
    ty: float,
    phi: float,
    theta: float,
    psi: float,
    convention: str = "zyx",
    fixed: bool = False,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a set of euler angles and an in-plane translation vector.

    Arguments
    ---------
    coords : `Array`, shape `(N, 3)`
        Coordinate system.
    tx : `float`
        In-plane translation in x direction.
    ty : `float`
        In-plane translation in y direction.
    phi : `float`
        Roll angle, ranging :math:`(-\pi, \pi]`.
    theta : `float`
        Pitch angle, ranging :math:`(0, \pi]`.
    psi : `float`
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    kwargs :
        Keyword arguments passed to ``make_rpy_rotation``

    Returns
    -------
    transformed : `Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = make_rpy_rotation(
        phi, theta, psi, convention=convention, fixed=fixed
    )
    translation = jnp.array([tx, ty, 0.0])
    transformation = SE3.from_rotation_and_translation(rotation, translation)
    transformed = jax.vmap(transformation.apply)(coords)

    return transformed


@jax.jit
def rotate_and_translate_wxyz(
    coords: Array,
    tx: float,
    ty: float,
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a quaternion and an in-plane translation vector.

    Arguments
    ---------
    coords : `Array` shape `(N, 3)`
        Coordinate system.
    tx : `float`
        In-plane translation in x direction.
    ty : `float`
        In-plane translation in y direction.
    qw : `float`
    qx : `float`
    qy : `float`
    qz : `float`

    Returns
    -------
    transformed : `Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    wxyz_xyz = jnp.array([qw, qx, qy, qz, tx, ty, 0.0])
    transformation = SE3(wxyz_xyz=wxyz_xyz)
    transformed = jax.vmap(transformation.apply)(coords)

    return transformed


@partial(jax.jit, static_argnames=["convention", "fixed"])
def rotate_rpy(
    coords: Array,
    phi: float,
    theta: float,
    psi: float,
    convention: str = "zyx",
    fixed: bool = False,
) -> Array:
    r"""
    Compute a coordinate rotation from
    a set of euler angles.

    Arguments
    ---------
    coords : `Array`, shape `(N, 3)`
        Coordinate system.
    phi : `float`
        Roll angle, ranging :math:`(-\pi, \pi]`.
    theta : `float`
        Pitch angle, ranging :math:`(0, \pi]`.
    psi : `float`
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    kwargs :
        Keyword arguments passed to ``make_rpy_rotation``

    Returns
    -------
    transformed : `Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = make_rpy_rotation(
        phi, theta, psi, convention=convention, fixed=fixed
    )
    transformed = jax.vmap(rotation.apply)(coords)

    return transformed


@jax.jit
def rotate_wxyz(
    coords: Array,
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> Array:
    r"""
    Compute a coordinate rotation from a quaternion.

    Arguments
    ---------
    coords : `Array` shape `(N, 3)`
        Coordinate system.
    qw : `float`
    qx : `float`
    qy : `float`
    qz : `float`

    Returns
    -------
    transformed : `Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """

    wxyz = jnp.array([qw, qx, qy, qz])
    rotation = SO3.from_quaternion_xyzw(wxyz)
    transformed = jax.vmap(rotation.apply)(coords)

    return transformed


@jax.jit
def shift_phase(
    density: Array,
    coords: Array,
    tx: float,
    ty: float,
) -> Array:
    r"""
    Compute the phase shifted density field from
    an in-plane real space translation.

    Arguments
    ---------
    density : `Array` shape `(N)`
        Coordinate system.
    coords : `Array` shape `(N, 3)`
        Coordinate system.
    tx : `float`
        In-plane translation in x direction.
    ty : `float`
        In-plane translation in y direction.

    Returns
    -------
    transformed : `Array`, shape `(N,)`
        Rotated and translated coordinate system.
    """
    xyz = jnp.array([tx, ty, 0.0])
    shift = jnp.exp(1.0j * 2 * jnp.pi * jnp.matmul(coords, xyz))
    transformed = density * shift

    return transformed


def make_rpy_rotation(
    phi: float,
    theta: float,
    psi: float,
    convention: str = "zyx",
    fixed: bool = False,
) -> SO3:
    """
    Helper routine to generate a rotation in a particular
    convention.
    """
    # Convert theta from [-pi/2, pi/2] to [0, pi]
    theta += jnp.pi / 2
    # Generate sequence of rotations
    rotations = [getattr(SO3, f"from_{axis}_radians") for axis in convention]
    # Gather set of angles (flip psi to match cisTEM)
    angles = [phi, theta, -psi] if fixed else [-psi, theta, phi]
    rotation = (
        rotations[0](angles[0])
        @ rotations[1](angles[1])
        @ rotations[2](angles[2])
    )

    return rotation

"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = [
    "rotate_and_translate_rpy",
    "rotate_and_translate_wxyz",
    "Pose",
    "EulerPose",
    "QuaternionPose",
]

from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jaxlie import SE3, SO3

from ..core import Array, Scalar, dataclass, Serializable


@dataclass
class Pose(Serializable, metaclass=ABCMeta):
    """
    Base class PyTree container for the image pose.

    Subclasses should choose a viewing convention,
    such as with Euler angles or Quaternions. In particular,

        1) Define angular coordinates
        2) Overwrite the ``Pose.transform`` method.
        3) Use the ``jax_2dtm.types.dataclass`` decorator.

    Attributes
    ----------`
    offset_x : `jax_2dtm.types.Scalar`
        In-plane translations in x direction.
    offset_y : `jax_2dtm.types.Scalar`
        In-plane translations in y direction.
    """

    offset_x: Scalar = 0.0
    offset_y: Scalar = 0.0

    @abstractmethod
    def transform(coordinates: Array) -> Array:
        """Transformation method for a particular pose convention."""
        raise NotImplementedError


@dataclass
class EulerPose(Pose):
    """
    An image pose using Euler angles.

    Attributes
    ----------
    view_phi : `jax_2dtm.types.Scalar`
        Roll angles, ranging :math:`(-\pi, \pi]`.
    view_theta : `jax_2dtm.types.Scalar`
        Pitch angles, ranging :math:`(0, \pi]`.
    view_psi : `jax_2dtm.types.Scalar`
        Yaw angles, ranging :math:`(-\pi, \pi]`.
    """

    view_phi: Scalar = 0.0
    view_theta: Scalar = 0.0
    view_psi: Scalar = 0.0

    def transform(self, coordinates: Array) -> Array:
        """Transform coordinates from a set of Euler angles."""
        N = np.prod(coordinates.shape[:-1])
        return rotate_and_translate_rpy(
            coordinates.reshape((N, 3)), *self.iter_data()
        ).reshape(coordinates.shape)


@dataclass
class QuaternionPose(Pose):
    """
    An image pose using unit Quaternions.

    Attributes
    ----------
    view_qx : `jax_2dtm.types.Scalar`

    view_qy : `jax_2dtm.types.Scalar`

    view_qz : `jax_2dtm.types.Scalar`

    """

    view_qw: Scalar = 1.0
    view_qx: Scalar = 0.0
    view_qy: Scalar = 0.0
    view_qz: Scalar = 0.0

    def transform(self, coordinates: Array) -> Array:
        """Transform coordinates from an offset and unit quaternion."""
        N = np.prod(coordinates.shape[:-1])
        return rotate_and_translate_wxyz(
            coordinates.reshape((N, 3)), *self.iter_data()
        ).reshape(coordinates.shape)


@jax.jit
def rotate_and_translate_rpy(
    coords: Array,
    tx: float,
    ty: float,
    phi: float,
    theta: float,
    psi: float,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a set of euler angles and an in-plane translation vector.

    Arguments
    ---------
    coords : `jax.Array`, shape `(N, 3)`
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

    Returns
    -------
    transformed : `jax.Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = SO3.from_rpy_radians(phi, theta, psi)
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
    coords : `jax.Array` shape `(N, 3)`
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
    transformed : `jax.Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    wxyz_xyz = jnp.array([qw, qx, qy, qz, tx, ty, 0.0])
    transformation = SE3(wxyz_xyz=wxyz_xyz)
    transformed = jax.vmap(transformation.apply)(coords)

    return transformed


@jax.jit
def rotate(
    coords: Array,
    phi: float,
    theta: float,
    psi: float,
) -> Array:
    r"""
    Compute a coordinate rotation from
    a set of euler angles.

    Arguments
    ---------
    coords : `jax.Array`, shape `(N, 3)`
        Coordinate system.
    phi : `float`
        Roll angle, ranging :math:`(-\pi, \pi]`.
    theta : `float`
        Pitch angle, ranging :math:`(0, \pi]`.
    psi : `float`
        Yaw angle, ranging :math:`(-\pi, \pi]`.

    Returns
    -------
    transformed : `jax.Array`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = SO3.from_rpy_radians(phi, theta, psi)
    transformed = jax.vmap(rotation.apply)(coords)

    return transformed

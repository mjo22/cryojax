"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = ["rotate_and_translate", "Pose", "Cloud"]

import jax.numpy as jnp
from jax import vmap, jit
from jaxlie import SE3, SO3
from ..types import Array, Scalar, dataclass, field


@dataclass
class Pose:
    """
    Attributes
    ----------
    view_phi : Scalar, `float` or shape `(M,)`
        Roll angles, ranging :math:`(-\pi, \pi]`.
    view_theta : Scalar, `float` or shape `(M,)`
        Pitch angles, ranging :math:`(0, \pi]`.
    view_psi : Scalar, `float` or shape `(M,)`
        Yaw angles, ranging :math:`(-\pi, \pi]`.
    offset_x : Scalar, `float` or shape `(M,)`
        In-plane translations in x direction.
    offset_y : Scalar, `float` or shape `(M,)`
        In-plane translations in y direction.
    """

    view_phi: Scalar
    view_theta: Scalar
    view_psi: Scalar
    offset_x: Scalar
    offset_y: Scalar


@dataclass
class Cloud:
    """
    Attributes
    ----------
    density : ArrayLike, shape `(N,)`
        3D electron density cloud.
    coordinates : ArrayLike, shape `(N, 3)`
        Cartesian coordinate system for density cloud.
    box_size : shape `(3,)`
        3D cartesian box that ``coords`` lies in. This
        should have dimensions of length.
    """

    density: Array = field(pytree_node=False)
    coordinates: Array = field(pytree_node=False)
    box_size: Array = field(pytree_node=False)


def rotate_and_translate(cloud: Cloud, pose: Pose) -> Cloud:
    """
    Compute an SE3 transformation of a point cloud,
    by an imaging pose, considering only in-plane translations.

    Arguments
    ---------
    cloud :
        3D electron density point cloud.
    pose :
        Imaging pose.
    """
    transformed_coords = _rotate_and_translate(
        cloud.coordinates, *pose.iter_data()
    )
    transformed_cloud = Cloud(
        cloud.density, transformed_coords, cloud.box_size
    )

    return transformed_cloud


@jit
def _rotate_and_translate(
    coords: Array,
    phi: Scalar,
    theta: Scalar,
    psi: Scalar,
    t_x: Scalar,
    t_y: Scalar,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a set of euler angles and an in-plane translation vector.

    Arguments
    ---------
    coords : shape `(N, 3)`
        Coordinate system.
    phi :
        Roll angle, ranging :math:`(-\pi, \pi]`.
    theta :
        Pitch angle, ranging :math:`(0, \pi]`.
    psi :
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    t_x :
        In-plane translation in x direction.
    t_y :
        In-plane translation in y direction.

    Returns
    -------
    transformed : shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = SO3.from_rpy_radians(phi, theta, psi)
    translation = jnp.array([t_x, t_y, 0.0])
    transformation = SE3.from_rotation_and_translation(rotation, translation)
    transformed = vmap(transformation.apply)(coords)

    return transformed


@jit
def _rotate(
    coords: Array,
    phi: Scalar,
    theta: Scalar,
    psi: Scalar,
) -> Array:
    r"""
    Compute a coordinate rotation from
    a set of euler angles.

    Arguments
    ---------
    coords : shape `(N, 3)`
        Coordinate system.
    phi :
        Roll angle, ranging :math:`(-\pi, \pi]`.
    theta :
        Pitch angle, ranging :math:`(0, \pi]`.
    psi :
        Yaw angle, ranging :math:`(-\pi, \pi]`.

    Returns
    -------
    transformed : shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = SO3.from_rpy_radians(phi, theta, psi)
    transformed = vmap(rotation.apply)(coords)

    return transformed

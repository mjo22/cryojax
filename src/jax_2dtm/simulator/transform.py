"""
Routines that compute coordinate rotations and translations.
"""

__all__ = ["rotate", "rotate_and_translate"]

import jax.numpy as jnp
from jax import vmap, jit
from jaxlie import SE3, SO3
from jax_2dtm.types import Array, Scalar


@jit
def rotate(
    coords: Array,
    phi: Scalar,
    theta: Scalar,
    psi: Scalar,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a wxyz quaternion and xyz translation vector.

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


@jit
def rotate_and_translate(
    coords: Array,
    phi: Scalar,
    theta: Scalar,
    psi: Scalar,
    t_x: Scalar,
    t_y: Scalar,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a wxyz quaternion and xyz translation vector.

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

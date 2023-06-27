"""
Routines that compute coordinate rotations and translations.
"""

__all__ = ["rotate_and_translate"]

import jax.numpy as jnp
from jax import vmap, jit
from jaxlie import SE3, SO3
from jax_2dtm.types import Array


@jit
def rotate_and_translate(
    coords: Array,
    view_phi: float,
    view_theta: float,
    view_psi: float,
    x_offset: float,
    y_offset: float,
) -> Array:
    r"""
    Compute a coordinate rotation and translation from
    a wxyz quaternion and xyz translation vector.

    Arguments
    ---------
    coords : shape `(N, 3)`
        Coordinate system.
    view_phi :
        Roll angle, ranging :math:`(-\pi, \pi]`.
    view_theta :
        Pitch angle, ranging :math:`(0, \pi]`.
    view_psi :
        Yaw angle, ranging :math:`(-\pi, \pi]`.
    x_offset :
        In-plane translation in x direction.
    y_offset :
        In-plane translation in y direction.

    Returns
    -------
    transformed : shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    rotation = SO3.from_rpy_radians(view_phi, view_theta, view_psi)
    translation = jnp.array([x_offset, y_offset, 0.0])
    transformation = SE3.from_rotation_and_translation(rotation, translation)
    transformed = vmap(transformation.apply)(coords)

    return transformed

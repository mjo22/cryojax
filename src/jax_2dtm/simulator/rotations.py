#!/usr/bin/env python3
"""
Routines that compute rotations and translataions of microtubule protofilaments.
"""

__all__ = ["rotate_wxyz", "rotate_rpy", "rotate_and_translate"]

import jax.numpy as jnp
from jax import vmap, jit
from jaxlie import SE3, SO3


@jit
def rotate_wxyz(coords, wxyz):
    """
    Compute a coordinate rotation from a wxyz quaternion.

    Arguments
    ---------
    coords : `jnp.ndarray`, shape `(N, 3)`
        Coordinate system.
    wxyz : `jnp.ndarray`, shape `(4,)`
        Quaternion for coordinate rotation.

    Returns
    -------
    transformed : `jnp.ndarray`, shape `(N, 3)`
        Rotated coordinate system.
    """
    T = SO3(wxyz=wxyz)
    transformed = vmap(T.apply)(coords)

    return transformed


@jit
def rotate_rpy(coords, rpy):
    """
    Compute a coordinate rotation from roll, pitch,
    and yaw euler angles.

    Arguments
    ---------
    coords : `jnp.ndarray`, shape `(N, 3)`
        Coordinate system.
    rpy : `jnp.ndarray`, shape `(3,)`
        Roll, pitch, and yaw for coordinate rotation.

    Returns
    -------
    transformed : `jnp.ndarray`, shape `(N, 3)`
        Rotated coordinate system.
    """
    T = SO3.from_rpy_radians(*rpy)
    transformed = vmap(T.apply)(coords)

    return transformed


@jit
def rotate_and_translate(coords, wxyz_xyz):
    """
    Compute a coordinate rotation and translation from
    a wxyz quaternion and xyz translation vector.

    Arguments
    ---------
    coords : `jnp.ndarray`, shape `(N, 3)`
        Coordinate system.
    wxyz_xyz : `jnp.ndarray`, shape `(7,)`
        wxyz Quaternion concatenated with xyz translation vector.

    Returns
    -------
    transformed : `jnp.ndarray`, shape `(N, 3)`
        Rotated and translated coordinate system.
    """
    T = SE3(wxyz_xyz=wxyz_xyz)
    transformed = vmap(T.apply)(coords)

    return transformed


if __name__ == "__main__":
    from jax import jit
    from jax_2dtm.io import load_mrc
    from coordinates import coordinatize

    template = read_mrc("./example/6dpu_14pf_bfm1_ps1_1.mrc")
    shape = template.shape

    model, coords = coordinatize(template)

    rpy = jnp.array([1.0, 0.0, 0.0])
    rotate_jit = jit(rotate_rpy)
    rotated_coords = rotate_jit(coords, rpy)

    print(coords, rotated_coords)

"""
Routines that compute coordinate rotations and translations.
"""

from __future__ import annotations

__all__ = ["rotate_and_translate", "Pose", "Cloud"]

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import vmap, jit
from jaxlie import SE3, SO3

from .scattering import project_with_nufft
from ..types import Array, Scalar, dataclass, field

if TYPE_CHECKING:
    from .image import ImageConfig


@dataclass
class Pose:
    """
    PyTree container for the image pose.

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

    view_phi: Scalar = 0.0
    view_theta: Scalar = 0.0
    view_psi: Scalar = 0.0
    offset_x: Scalar = 0.0
    offset_y: Scalar = 0.0


@dataclass
class Cloud:
    """
    Abstraction of a 3D electron density point cloud.

    Attributes
    ----------
    density : Array, shape `(N,)`
        3D electron density cloud.
    coordinates : Array, shape `(N, 3)`
        Cartesian coordinate system for density cloud.
    box_size : Array, shape `(3,)`
        3D cartesian box that ``coordinates`` lie in. This
        should have dimensions of length.
    """

    density: Array = field(pytree_node=False)
    coordinates: Array = field(pytree_node=False)
    box_size: Array = field(pytree_node=False)

    def view(self, pose: Pose) -> Cloud:
        """
        Compute an SE3 transformation of a point cloud,
        by an imaging pose, considering only in-plane translations.

        Arguments
        ---------
        pose :
            Imaging pose.
        """
        coordinates = rotate_and_translate(self.coordinates, *pose.iter_data())

        return self.replace(coordinates=coordinates)

    def project(self, config: ImageConfig) -> Array:
        """
        Compute projection of the point cloud onto
        an imaging plane.

        Arguments
        ---------
        config :
            The image configuration.
        """

        return project_with_nufft(config, *self.iter_meta())


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
def rotate(
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

"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = ["Cloud"]

from .pose import Pose
from .scattering import ScatteringConfig
from ..types import Array, dataclass, field


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
        pose : `jax_2dtm.simulator.Pose`
            The imaging pose.
        """
        coordinates = pose.transform(self.coordinates)

        return self.replace(coordinates=coordinates)

    def project(self, scattering: ScatteringConfig) -> Array:
        """
        Compute projection of the point cloud onto
        an imaging plane.

        Arguments
        ---------
        scattering : `jax_2dtm.simulator.ScatteringConfig`
            The scattering configuration. This is an
            ``ImageConfig``, subclassed to include a scattering
            routine.
        """

        return scattering.project(*self.iter_meta())

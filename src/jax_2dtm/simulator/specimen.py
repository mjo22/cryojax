"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = ["Specimen", "ElectronDensity", "ElectronCloud", "ElectronGrid"]

from abc import ABCMeta, abstractmethod

from .pose import Pose
from ..core import Array, dataclass, field, Serializable
from . import ScatteringConfig


@dataclass
class Specimen(Serializable, metaclass=ABCMeta):
    """
    Abstraction of a biological specimen.
    """

    @abstractmethod
    def view(self, pose: Pose) -> Specimen:
        """
        View the specimen at a given pose.

        Arguments
        ---------
        pose : `jax_2dtm.simulator.Pose`
            The imaging pose.
        """
        raise NotImplementedError

    @abstractmethod
    def scatter(self, scattering: ScatteringConfig) -> Array:
        """
        Compute the scattered wave of the specimen in the
        exit plane.

        Arguments
        ---------
        scattering : `jax_2dtm.simulator.ScatteringConfig`
            The scattering configuration. This is an
            ``ImageConfig``, subclassed to include a scattering
            routine.
        """
        raise NotImplementedError


@dataclass
class ElectronDensity(Specimen):
    """
    Electron density contrast representation of a specimen.

    Attributes
    ----------
    density : `Array`
    coordinates : `Array`
    box_size : `Array`, shape `(3,)`
    """

    density: Array = field(pytree_node=False)
    coordinates: Array = field(pytree_node=False)
    box_size: Array = field(pytree_node=False)

    def scatter(self, scattering: ScatteringConfig) -> Array:
        """
        Compute the 2D rendering of the point cloud in the
        object plane.
        """

        return scattering.scatter(*self.iter_meta()[:3])


@dataclass
class ElectronCloud(ElectronDensity):
    """
    Abstraction of a 3D electron density point cloud.

    Attributes
    ----------
    density : `Array`, shape `(N,)`
        3D electron density cloud.
    coordinates : `Array`, shape `(N, 3)`
        Cartesian coordinate system for density cloud.
    box_size : `Array`, shape `(3,)`
        3D cartesian  that ``coordinates`` lie in. This
        should have dimensions of length.
    """

    def view(self, pose: Pose) -> ElectronCloud:
        """
        Compute rotations and translations of a point cloud,
        by an imaging pose, considering only in-plane translations.

        This transformation will rotate and translate the density
        cloud's coordinates.
        """
        _, coordinates = pose.transform(
            self.density, self.coordinates, real=True
        )

        return self.replace(coordinates=coordinates)


@dataclass
class ElectronGrid(ElectronDensity):
    """
    Abstraction of a 3D electron density voxel grid.

    The voxel grid is assumed to be in Fourier space.

    Attributes
    ----------
    density : `Array`, shape `(N1, N2, N3)`
        3D electron density grid.
    coordinates : `Array`, shape `(N1, N2, N3, 3)`
        Cartesian coordinate system for density grid.
    box_size : `Array`, shape `(3,)`
        3D cartesian  that ``coordinates`` lie in. This
        should have dimensions of length.
    """

    def view(self, pose: Pose) -> ElectronGrid:
        """
        Compute rotations and translations of a point cloud,
        by an imaging pose, considering only in-plane translations.

        This transformation will rotate the coordinates and phase shift
        the density.
        """
        # Flatten density and coordinates
        N1, N2, N3 = self.coordinates.shape[:-1]
        N = N1 * N2 * N3
        density = self.density.reshape((N,))
        coordinates = self.coordinates.reshape((N, 3))
        # Transform
        density, coordinates = pose.transform(
            self.density, coordinates, real=False
        )
        # Reshape back to voxel grid
        density = density.reshape(self.density.shape)
        coordinates = coordinates.reshape(self.coordinates.shape)

        return self.replace(density=density, coordinates=coordinates)

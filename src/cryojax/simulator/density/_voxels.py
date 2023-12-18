"""
Voxel-based electron density representations.
"""

__all__ = ["Voxels", "VoxelCloud", "VoxelGrid"]

import equinox as eqx

from abc import abstractmethod
from typing import Any, Type
from jaxtyping import Float, Array
from equinox import AbstractVar

from ._density import ElectronDensity
from ..pose import Pose
from ...io import load_voxel_cloud, load_fourier_grid
from ...core import field
from ...typing import (
    ComplexVolume,
    RealCloud,
    CloudCoords3D,
)

_VolumeSliceCoords = Float[Array, "N1 N2 1 3"]


class Voxels(ElectronDensity):
    """
    Voxel-based electron density representation.

    Attributes
    ----------
    weights :
        The electron density.
    coordinates :
        The coordinate system.
    """

    weights: AbstractVar[Array]
    coordinates: AbstractVar[Array]

    @classmethod
    def from_file(
        cls: Type["Voxels"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "Voxels":
        """Load a ElectronDensity."""
        return cls.from_mrc(filename, config=config, **kwargs)

    @classmethod
    @abstractmethod
    def from_mrc(
        cls: Type["Voxels"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "Voxels":
        """Load a ElectronDensity from MRC file format."""
        raise NotImplementedError


class VoxelGrid(Voxels):
    """
    Abstraction of a 3D electron density voxel grid.

    The voxel grid should be given in Fourier space.

    Attributes
    ----------
    weights :
        3D electron density grid in Fourier space.
    coordinates : shape `(N1, N2, 1, 3)`
        Central slice of cartesian coordinate system.
    """

    weights: ComplexVolume = field()
    coordinates: _VolumeSliceCoords = field()

    real: bool = field(default=False, static=True)

    def __check_init__(self):
        if self.real is True:
            raise NotImplementedError(
                "Real voxel grid densities are not supported."
            )

    def rotate_to(self, pose: Pose) -> "VoxelGrid":
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        coordinates = pose.rotate(self.coordinates, real=self.real)

        return eqx.tree_at(lambda d: d.coordinates, self, coordinates)

    @classmethod
    def from_mrc(
        cls: Type["VoxelGrid"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "VoxelGrid":
        """
        See ``cryojax.io.voxel.load_fourier_grid`` for
        documentation.
        """
        return cls(**load_fourier_grid(filename, **config), **kwargs)


class VoxelCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    The point cloud is given in real space.

    Attributes
    ----------
    weights :
        3D electron density cloud.
    coordinates :
        Cartesian coordinate system for density cloud.
    """

    weights: RealCloud = field()
    coordinates: CloudCoords3D = field()

    real: bool = field(default=True, static=True)

    def __check_init__(self):
        if self.real is False:
            raise NotImplementedError(
                "Fourier voxel cloud densities are not supported."
            )

    def rotate_to(self, pose: Pose) -> "VoxelCloud":
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        coordinates = pose.rotate(self.coordinates, real=self.real)

        return eqx.tree_at(lambda d: d.coordinates, self, coordinates)

    @classmethod
    def from_mrc(
        cls: Type["VoxelCloud"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "VoxelCloud":
        """
        See ``cryojax.io.voxel.load_grid_as_cloud`` for
        documentation.
        """
        return cls(**load_voxel_cloud(filename, **config), **kwargs)

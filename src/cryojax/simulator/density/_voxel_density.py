"""
Voxel-based electron density representations.
"""

__all__ = ["Voxels", "VoxelCloud", "VoxelGrid"]

import equinox as eqx

from abc import abstractmethod
from typing import Any, Type, ClassVar
from jaxtyping import Float, Array
from equinox import AbstractVar

from ._electron_density import ElectronDensity
from ..pose import Pose
from ...io import load_voxel_cloud, load_fourier_grid
from ...core import field
from ...typing import Real_, RealCloud, CloudCoords3D

_CubicVolume = Float[Array, "N N N"]
_VolumeSliceCoords = Float[Array, "N N 1 3"]


class Voxels(ElectronDensity):
    """
    Voxel-based electron density representation.

    Attributes
    ----------
    weights :
        The electron density.
    coordinates :
        The coordinate system.
    voxel_size
        The voxel size of the electron density.
    """

    weights: AbstractVar[Array]
    coordinates: AbstractVar[Array]
    voxel_size: Real_ = field()

    @classmethod
    def from_file(
        cls: Type["Voxels"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "Voxels":
        """Load a ElectronDensity."""
        return cls.from_mrc(
            filename, config=config, _is_stacked=False, **kwargs
        )

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
    coordinates :
        Central slice of cartesian coordinate system.
    """

    weights: _CubicVolume = field()
    coordinates: _VolumeSliceCoords = field()

    is_real: ClassVar[bool] = False

    def rotate_to(self, pose: Pose) -> "VoxelGrid":
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        coordinates = pose.rotate(self.coordinates, is_real=self.is_real)

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

    is_real: ClassVar[bool] = True

    def rotate_to(self, pose: Pose) -> "VoxelCloud":
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        coordinates = pose.rotate(self.coordinates, is_real=self.is_real)

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

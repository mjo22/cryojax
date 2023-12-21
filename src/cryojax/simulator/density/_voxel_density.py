"""
Voxel-based electron density representations.
"""

__all__ = ["Voxels", "VoxelCloud", "VoxelGrid"]

import equinox as eqx

from abc import abstractmethod
from typing import Any, Type
from jaxtyping import Float, Array
from equinox import AbstractVar

import jax.numpy as jnp

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
            filename, config=config, is_stacked=False, **kwargs
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

    @classmethod
    def from_stack(cls: Type["Voxels"], stack: list["Voxels"]) -> "Voxels":
        """
        Stack a list of electron densities along the leading
        axis of a single electron density.
        """
        if not all([cls == type(density) for density in stack]):
            raise TypeError(
                "Electron density stack should all be of the same type."
            )
        if not all([stack[0].is_real == density.is_real for density in stack]):
            raise TypeError(
                "Electron density stack should all be in real or fourier space."
            )
        weights = jnp.stack([density.weights for density in stack], axis=0)
        coordinates = jnp.stack(
            [density.coordinates for density in stack], axis=0
        )
        voxel_size = jnp.stack(
            [density.voxel_size for density in stack], axis=0
        )
        return cls(
            weights=weights,
            coordinates=coordinates,
            voxel_size=voxel_size,
            is_real=stack[0].is_real,
            is_stacked=True,
        )

    def __getitem__(self, idx: int) -> "Voxels":
        if self.is_stacked:
            cls = type(self)
            return cls(
                weights=self.weights[idx],
                coordinates=self.coordinates[idx],
                voxel_size=self.voxel_size[idx],
                is_real=self.is_real,
                is_stacked=False,
            )
        else:
            return self

    def __len__(self) -> int:
        if self.is_stacked:
            return self.weights.shape[0]
        else:
            return 1


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

    is_real: bool = field(default=False, static=True, kw_only=True)

    def __check_init__(self):
        if self.is_real is True:
            raise NotImplementedError(
                "Real voxel grid densities are not supported."
            )

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

    is_real: bool = field(default=True, static=True, kw_only=True)

    def __check_init__(self):
        if self.is_real is False:
            raise NotImplementedError(
                "Fourier voxel cloud densities are not supported."
            )

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

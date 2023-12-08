"""
Electron density representations.
"""

from __future__ import annotations

__all__ = [
    "ElectronDensity",
    "Voxels",
    "ElectronCloud",
    "ElectronGrid",
]

import equinox as eqx

from abc import abstractmethod
from typing import Optional, Any, Type
from jaxtyping import Array

from .scattering import ScatteringConfig
from .pose import Pose
from ..io import load_grid_as_cloud, load_fourier_grid
from ..core import field, Module
from ..types import (
    Real_,
    ComplexImage,
    ComplexVolume,
    VolumeCoords,
    RealCloud,
    CloudCoords,
)


class ElectronDensity(Module):
    """
    Abstraction of an electron density map.
    """

    @abstractmethod
    def view(self, pose: Pose) -> ElectronDensity:
        """
        View the electron density at a given pose.

        Arguments
        ---------
        pose :
            The imaging pose.
        """
        raise NotImplementedError

    @abstractmethod
    def scatter(
        self, scattering: ScatteringConfig, resolution: Real_
    ) -> ComplexImage:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        scattering :
            The scattering configuration. This is an
            ``ImageConfig``, subclassed to include a scattering
            routine.
        resolution :
            The rasterization resolution.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(
        cls: Type[ElectronDensity],
        filename: str,
        config: Optional[dict] = None,
    ) -> ElectronDensity:
        """
        Load an ElectronDensity from a file.

        This method should be used to instantiate and
        deserialize ElectronDensity.
        """
        raise NotImplementedError


class Voxels(ElectronDensity):
    """
    Voxel-based electron density contrast representation
    of an electron density.

    Attributes
    ----------
    weights :
        The density contrast.
    coordinates :
        The coordinate system.
    """

    weights: Array = field()
    coordinates: Array = field()

    def scatter(
        self, scattering: ScatteringConfig, resolution: Real_
    ) -> ComplexImage:
        """
        Compute the 2D rendering of the point cloud in the
        object plane.
        """
        return scattering.scatter(self.weights, self.coordinates, resolution)

    @classmethod
    def from_file(
        cls: Type[Voxels],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> Voxels:
        """Load a ElectronDensity."""
        return cls.from_mrc(filename, config=config, **kwargs)

    @classmethod
    @abstractmethod
    def from_mrc(
        cls: Type[Voxels],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> Voxels:
        """Load a ElectronDensity from MRC file format."""
        raise NotImplementedError


class ElectronCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    Attributes
    ----------
    weights :
        3D electron density cloud.
    coordinates :
        Cartesian coordinate system for density cloud.
    """

    weights: RealCloud = field()
    coordinates: CloudCoords = field()

    def view(self, pose: Pose) -> ElectronCloud:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        coordinates = pose.rotate(self.coordinates, real=True)

        return eqx.tree_at(lambda d: d.coordinates, self, coordinates)

    @classmethod
    def from_mrc(
        cls: Type[ElectronCloud],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> ElectronCloud:
        """
        See ``cryojax.io.voxel.load_grid_as_cloud`` for
        documentation.
        """
        return cls(**load_grid_as_cloud(filename, **config), **kwargs)


class AtomCloud(ElectronDensity):
    """
    Abstraction of a point cloud of atoms.
    """

    density: Array = field()
    coordinates: Array = field()
    variances: Array = field()
    identity: Array = field()

    def scatter(
        self, scattering: ScatteringConfig, resolution: Real_
    ) -> ComplexImage:
        """
        Compute the 2D rendering of the point cloud in the
        object plane.
        """

        return scattering.scatter(
            self.density,
            self.coordinates,
            resolution,
            self.identity,
            self.variances,
        )

    def view(self, pose: Pose) -> AtomCloud:
        coordinates = pose.rotate(self.coordinates, real=True)
        return eqx.tree_at(lambda d: d.coordinates, self, coordinates)

    @classmethod
    def from_file(
        cls: Type[Voxels],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> Voxels:
        """
        Load an Atom Cloud

        TODO: What is the file format appropriate here? Q. for Michael...
        """
        raise NotImplementedError
        # return cls.from_mrc(filename, config=config, **kwargs)


class ElectronGrid(Voxels):
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
    coordinates: VolumeCoords = field()

    def view(self, pose: Pose) -> ElectronGrid:
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        coordinates = pose.rotate(self.coordinates, real=False)

        return eqx.tree_at(lambda d: d.coordinates, self, coordinates)

    @classmethod
    def from_mrc(
        cls: Type[ElectronGrid],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> ElectronGrid:
        """
        See ``cryojax.io.voxel.load_fourier_grid`` for
        documentation.
        """
        return cls(**load_fourier_grid(filename, **config), **kwargs)

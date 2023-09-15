"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = [
    "ElectronDensity",
    "Voxels",
    "ElectronCloud",
    "ElectronGrid",
]

from abc import ABCMeta, abstractmethod
from typing import Optional, Any, Type
from functools import partial
from dataclasses import fields

from .scattering import ScatteringConfig
from .pose import Pose
from ..io import load_grid_as_cloud, load_fourier_grid
from ..core import Array, Parameter, dataclass, field, CryojaxObject


@partial(dataclass, kw_only=True)
class ElectronDensity(CryojaxObject, metaclass=ABCMeta):
    """
    Abstraction of an electron density map.

    Attributes
    ----------
    filename : `str`, optional
        The path to where the electron density is saved.
        This is required for deserialization and
        is used in ``ElectronDensity.from_file``.
    config : `dict`, optional
        The deserialization settings for
        ``ElectronDensity.from_file``.
    """

    # Fields configuring the file loader.
    filename: Optional[str] = field(pytree_node=False, default=None)
    config: dict = field(pytree_node=False, default_factory=dict)

    @abstractmethod
    def view(self, pose: Pose) -> ElectronDensity:
        """
        View the electron density at a given pose.

        Arguments
        ---------
        pose : `cryojax.simulator.Pose`
            The imaging pose.
        """
        raise NotImplementedError

    @abstractmethod
    def scatter(
        self, scattering: ScatteringConfig, resolution: Parameter
    ) -> Array:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration. This is an
            ``ImageConfig``, subclassed to include a scattering
            routine.
        resolution : `cryojax.core.Parameter`
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

    @classmethod
    def from_dict(
        cls: Type[ElectronDensity], kvs: dict, *, infer_missing: bool = False
    ) -> ElectronDensity:
        """
        Load a ``ElectronDensity`` from a dictionary. This function overwrites
        ``cryojax.core.Serializable.from_dict`` in order to
        avoid saving the large arrays typically stored in ``ElectronDensity``.
        """
        # Get fields that we want to decode
        fs = fields(cls)
        encoded = [
            f.name
            for f in fs
            if (
                not "encode" in f.metadata or f.metadata["encode"] is not False
            )
            and f.name not in ["filename", "config"]
        ]
        updates = {k: kvs[k] for k in encoded}
        # Get filename and configuration for I/O
        filename = kvs["filename"]
        config = kvs["config"]
        return cls.from_file(filename, config=config, **updates)


@dataclass
class Voxels(ElectronDensity):
    """
    Voxel-based electron density contrast representation
    of an electron density.

    Attributes
    ----------
    weights : `Array`
        The density contrast.
    coordinates : `Array`
        The coordinate system.
    """

    # Fields describing the density map.
    weights: Array = field(pytree_node=False, encode=False)
    coordinates: Array = field(pytree_node=False, encode=False)

    def scatter(
        self, scattering: ScatteringConfig, resolution: Parameter
    ) -> Array:
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


@dataclass
class ElectronCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    Attributes
    ----------
    weights : `Array`, shape `(N,)`
        3D electron density cloud.
    coordinates : `Array`, shape `(N, 3)`
        Cartesian coordinate system for density cloud.
    """

    def view(self, pose: Pose) -> ElectronCloud:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        coordinates = pose.rotate(self.coordinates, real=True)

        return self.replace(coordinates=coordinates)

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


@dataclass
class ElectronGrid(Voxels):
    """
    Abstraction of a 3D electron density voxel grid.

    The voxel grid should be given in Fourier space.

    Attributes
    ----------
    weights : `Array`, shape `(N1, N2, N3)`
        3D electron density grid in Fourier space.
    coordinates : `Array`, shape `(N1, N2, 1, 3)`
        Central slice of cartesian coordinate system.
    """

    def view(self, pose: Pose) -> ElectronGrid:
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        coordinates = pose.rotate(self.coordinates, real=False)

        return self.replace(coordinates=coordinates)

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

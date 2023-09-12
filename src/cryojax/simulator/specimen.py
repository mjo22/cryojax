"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = ["Specimen", "Voxels", "ElectronCloud", "ElectronGrid"]

from abc import ABCMeta, abstractmethod
from typing import Optional, Any, Type
from functools import partial
from dataclasses import fields

from .scattering import ScatteringConfig
from .pose import Pose
from ..io import load_grid_as_cloud, load_fourier_grid
from ..core import Array, Parameter, dataclass, field, CryojaxObject


@partial(dataclass, kw_only=True)
class Specimen(CryojaxObject, metaclass=ABCMeta):
    """
    Abstraction of a biological specimen.

    Attributes
    ----------
    filename : `str`, optional
        The path to where the specimen is saved.
        This is required for deserialization and
        is used in ``Specimen.from_file``.
    config : `dict`, optional
        The deserialization settings for
        ``Specimen.from_file``.
    resolution : `float`
        Rasterization resolution.
        This is in dimensions of length.
    """

    # Fields configuring the file loader.
    filename: Optional[str] = field(pytree_node=False, default=None)
    config: dict = field(pytree_node=False, default_factory=dict)

    # The resolution of the specimen
    resolution: Parameter = field()

    @abstractmethod
    def view(self, pose: Pose) -> Specimen:
        """
        View the specimen at a given pose.

        Arguments
        ---------
        pose : `cryojax.simulator.Pose`
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
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration. This is an
            ``ImageConfig``, subclassed to include a scattering
            routine.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(
        cls: Type[Specimen],
        filename: str,
        config: Optional[dict] = None,
    ) -> Specimen:
        """
        Load a Specimen from a file.

        This method should be used to instantiate and deserialize Specimen.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls: Type[Specimen], kvs: dict, *, infer_missing: bool = False
    ) -> Specimen:
        """
        Load a ``Specimen`` from a dictionary. This function overwrites
        ``cryojax.core.Serializable.from_dict`` in order to
        avoid saving the large arrays typically stored in ``Specimen``.
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
class Voxels(Specimen):
    """
    Voxel-based electron density contrast representation of a specimen.

    Attributes
    ----------
    density : `Array`
        The density contrast.
    coordinates : `Array`
        The coordinate system.
    """

    # Fields describing the density map.
    density: Array = field(pytree_node=False, encode=False)
    coordinates: Array = field(pytree_node=False, encode=False)

    def scatter(self, scattering: ScatteringConfig) -> Array:
        """
        Compute the 2D rendering of the point cloud in the
        object plane.
        """
        return scattering.scatter(
            self.density, self.coordinates, self.resolution
        )

    @classmethod
    def from_file(
        cls: Type[Voxels],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> Voxels:
        """Load a Specimen."""
        return cls.from_mrc(filename, config=config, **kwargs)

    @classmethod
    @abstractmethod
    def from_mrc(
        cls: Type[Voxels],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> Voxels:
        """Load a Specimen from MRC file format."""
        raise NotImplementedError


@dataclass
class ElectronCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    Attributes
    ----------
    density : `Array`, shape `(N,)`
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
    density : `Array`, shape `(N1, N2, N3)`
        3D electron density grid in Fourier space.
    coordinates : `Array`, shape `(N1, N2, 1, 3)`
        Central slice of cartesian coordinate system.
    """

    def view(self, pose: Pose) -> ElectronGrid:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will rotate the coordinates with a
        backrotation
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

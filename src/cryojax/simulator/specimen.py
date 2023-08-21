"""
Routines for representing and operating on 3D point clouds.
"""

from __future__ import annotations

__all__ = ["Specimen", "ElectronDensity", "ElectronCloud", "ElectronGrid"]

from abc import ABCMeta, abstractmethod
from typing import Optional, Any, Type
from functools import partial

import jax.numpy as jnp

from .pose import Pose
from ..io import load_grid_as_cloud, load_fourier_grid
from ..core import Array, dataclass, field, CryojaxObject
from . import ScatteringConfig


@partial(dataclass, kw_only=True)
class Specimen(CryojaxObject, metaclass=ABCMeta):
    """
    Abstraction of a biological specimen.
    """

    # Fields configuring the file loader.
    filename: Optional[str] = field(pytree_node=False, default=None)
    config: dict = field(pytree_node=False, default_factory=dict)

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
        cls: Type[Specimen], filename: str, **kwargs: Any
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
        return cls.from_file(kvs["filename"], **kvs["config"])


@dataclass
class ElectronDensity(Specimen):
    """
    Electron density contrast representation of a specimen.

    This base class represents a voxel-based electron density.

    Attributes
    ----------
    density : `Array`
        The density contrast.
    coordinates : `Array`
        The coordinate system.
    voxel_size : `Array`, shape (3,)
        The voxel size of the electron density map.
    filename : `str`, optional
        The path to where the density is saved. This is required for
        deserialization.
    """

    # Fields describing the density map.
    density: Array = field(pytree_node=False, encode=False)
    coordinates: Array = field(pytree_node=False, encode=False)
    voxel_size: Array = field(pytree_node=False, encode=False)

    def scatter(self, scattering: ScatteringConfig) -> Array:
        """
        Compute the 2D rendering of the point cloud in the
        object plane.
        """
        return scattering.scatter(
            self.density, self.coordinates, self.voxel_size
        )

    @classmethod
    def from_file(
        cls: Type[ElectronDensity], *args: Any, **kwargs: Any
    ) -> ElectronDensity:
        """Load a Specimen."""
        return cls.from_mrc(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_mrc(
        cls: Type[ElectronDensity], filename: str, **kwargs: Any
    ) -> ElectronDensity:
        """Load a Specimen from MRC file format."""
        raise NotImplementedError


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

    @classmethod
    def from_mrc(
        cls: Type[ElectronCloud], filename: str, **kwargs: Any
    ) -> ElectronCloud:
        """
        See ``cryojax.io.voxel.load_grid_as_cloud`` for
        documentation.
        """
        return cls(**load_grid_as_cloud(filename, **kwargs))


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
    """

    def view(self, pose: Pose) -> ElectronGrid:
        """
        Compute rotations and translations of a point cloud,
        by an imaging pose, considering only in-plane translations.

        This transformation will rotate the coordinates and phase shift
        the density.
        """
        density, coordinates = pose.transform(
            self.density, self.coordinates, real=False
        )

        return self.replace(density=density, coordinates=coordinates)

    @classmethod
    def from_mrc(
        cls: Type[ElectronGrid], filename: str, **kwargs: Any
    ) -> ElectronGrid:
        """
        See ``cryojax.io.voxel.load_fourier_grid`` for
        documentation.
        """
        return cls(**load_fourier_grid(filename, **kwargs))

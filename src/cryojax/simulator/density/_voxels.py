"""
Voxel-based electron density representations.
"""

__all__ = ["Voxels", "VoxelCloud", "VoxelGrid"]

import equinox as eqx

from abc import abstractmethod
from typing import Any, Type, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from equinox import AbstractVar

from ._density import ElectronDensity
from ..pose import Pose
from ...io import load_voxel_cloud, load_fourier_grid
from ...core import field
from cryojax.utils import make_frequencies, make_coordinates
from cryojax.typing import (
    ComplexVolume,
    RealCloud,
    CloudCoords3D,
    RealVolume,
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

    def view(self, pose: Pose) -> "VoxelGrid":
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

    def view(self, pose: Pose) -> "VoxelCloud":
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

    @classmethod
    def from_pdb(
        cls: Type["VoxelCloud"],
        filename: str,
        config: dict = {},
        **kwargs: Any,
    ) -> "VoxelCloud":
        """
        Loads a PDB file as a VoxelCloud.  Uses the Gemmi library.
        Heavily based on a code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        raise NotImplementedError
        return cls(**load_voxel_cloud(filename, **config), **kwargs)

    @classmethod
    def from_atom_cloud(
        cls: Type["VoxelCloud"],
        atom_cloud: "AtomCloud",
        resolution: float,
        **kwargs: Any,
    ) -> "VoxelCloud":
        """
        Convert an AtomCloud to a VoxelCloud.

        Parameters
        ----------
        atom_cloud :
            The atom cloud to convert.
        resolution :
            The resolution of the voxel grid.
        """
        import gemmi


def _eval_atom_gaussian_in_3d(coordinates, pos, a: float, b: float) -> Array:
    """
    Evaluate a gaussian on a 3D grid.

    Parameters
    ----------
    coordinates : `Array`, shape `(N1, N2, N3, 3)`
        The coordinates of the grid.
    pos : `Array`, shape `(3,)`
        The center of the gaussian.
    a : `float`
        A scale factor.
    b : `float`
        The scale of the gaussian.

    Returns
    -------
    density : `Array`, shape `(N1, N2, N3)`
        The density of the gaussian on the grid.
    """
    invb = 4.0 * jnp.pi / b
    sq_distances = jnp.sum(invb * (coordinates - pos) ** 2, axis=-1)
    density = jnp.exp(-jnp.pi * sq_distances) * a * invb
    return density


def _eval_form_factor_in_3d(coordinates, pos, a: float, b: float) -> Array:
    eval_fxn = jax.vmap(_eval_atom_gaussian_in_3d, in_axes=(None, None, 0, 0))
    return jnp.sum(eval_fxn(coordinates, pos, a, b), axis=0)


_eval_all_form_factors = jax.vmap(
    _eval_form_factor_in_3d, in_axes=(None, 0, 0, 0)
)


def _build_voxels_from_atoms(
    atom_coords: Array,
    ff_a: Array,
    ff_b: Array,
    shape: Tuple[int, int, int],
    voxel_size: float = 1.0,
) -> dict[str, Any]:
    """
    Build a voxel representation of an atomic model.

    Parameters
    ----------
    coords : `Array`, shape `(N, 3)`
        The coordinates of the atoms.
    ff_a : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        Intensity values for each Gaussian in the atom
    ff_b : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        The inverse scale factors for each Gaussian in the atom

    Returns
    -------
    voxels : `dict`
        3D electron density in a 3D voxel grid representation.
        Instantiates a ``cryojax.simulator.ElectronGrid``
    """
    # Load density and coordinates
    coordinates_3d = make_coordinates(shape, voxel_size)

    density = _eval_all_form_factors(
        coordinates_3d, atom_coords, ff_a, ff_b
    )  # shape (N1, N2, N3, N)

    voxels = jnp.sum(density, axis=0)

    # Get central z slice
    coordinates = jnp.expand_dims(coordinates_3d[:, :, 0, :], axis=2)
    return dict(weights=voxels, coordinates=coordinates)

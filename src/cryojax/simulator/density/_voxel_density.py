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

import jax.numpy as jnp

from ._electron_density import ElectronDensity
from ..pose import Pose
from ...io import (
    load_voxel_cloud,
    load_fourier_grid,
    read_atomic_model_from_pdb,
    read_atomic_model_from_cif,
    get_scattering_info_from_gemmi_model,
)
from ...core import field
from cryojax.utils import (
    make_frequencies,
    make_coordinates,
    flatten_and_coordinatize,
    rfftn,
)
from cryojax.typing import (
    ComplexVolume,
    RealCloud,
    CloudCoords3D,
    RealVolume,
    Real_,
)

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
            _is_stacked=True,
        )

    def __getitem__(self, idx: int) -> "Voxels":
        if self._is_stacked:
            cls = type(self)
            return cls(
                weights=self.weights[idx],
                coordinates=self.coordinates[idx],
                voxel_size=self.voxel_size[idx],
                is_real=self.is_real,
                _is_stacked=False,
            )
        else:
            return self

    def __len__(self) -> int:
        if self._is_stacked:
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

    @classmethod
    def from_pdb(
        cls: Type["VoxelGrid"],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        **kwargs: Any,
    ) -> "VoxelGrid":
        """
        Loads a PDB file as a VoxelGrid.
        """
        model = read_atomic_model_from_pdb(filename)
        return cls.from_gemmi(model, n_voxels_per_side, voxel_size, **kwargs)

    @classmethod
    def from_cif(
        cls: Type["VoxelGrid"],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        **kwargs: Any,
    ) -> "VoxelGrid":
        """
        Loads a PDB file as a VoxelGrid.
        """
        model = read_atomic_model_from_cif(filename)
        return cls.from_gemmi(model, n_voxels_per_side, voxel_size, **kwargs)

    @classmethod
    def from_gemmi(
        cls: Type["VoxelGrid"],
        model,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        real: bool = False,
        **kwargs: Any,
    ) -> "VoxelGrid":
        """
        Loads a PDB file as a VoxelCloud.  Uses the Gemmi library.
        Heavily based on a code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        coords, a_vals, b_vals = get_scattering_info_from_gemmi_model(model)

        coordinates_3d = make_coordinates(n_voxels_per_side, voxel_size)
        density = _build_real_space_voxels_from_atoms(
            coords, a_vals, b_vals, coordinates_3d
        )

        if real:
            # Get the central z slice of the coordinate system
            z_plane_coordinates = jnp.expand_dims(
                coordinates_3d[:, :, n_voxels_per_side[-1] // 2, :], axis=2
            )
            vdict = {
                "weights": density,
                "coordinates": z_plane_coordinates,
                "voxel_size": voxel_size,
                "is_real": True,
            }
            return cls(**vdict, **kwargs)
        else:
            # Build the Fourier grid and take the central slice.
            fourier_space_density = rfftn(density)
            frequency_grid = make_frequencies(
                fourier_space_density.shape, voxel_size
            )

            z_plane_frequencies = jnp.expand_dims(
                frequency_grid[:, :, 0, :], axis=2
            )

            vdict = {
                "weights": fourier_space_density,
                "coordinates": z_plane_frequencies,
                "voxel_size": voxel_size,
                "is_real": False,
            }
            return cls(**vdict, **kwargs)


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

    @classmethod
    def from_pdb(
        cls: Type["VoxelCloud"],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        mask: bool = True,
        indexing: str = "xy",
        **kwargs: Any,
    ) -> "VoxelCloud":
        """
        Loads a PDB file as a VoxelCloud.  Uses the Gemmi library.
        Adapted from code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        model = read_atomic_model_from_pdb(filename)
        return cls.from_gemmi(
            model, n_voxels_per_side, voxel_size, mask, indexing, **kwargs
        )

    @classmethod
    def from_cif(
        cls: Type["VoxelCloud"],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        mask: bool = True,
        indexing: str = "xy",
        **kwargs: Any,
    ) -> "VoxelCloud":
        """
        Loads a PDB file as a VoxelCloud.  Uses the Gemmi library.
        Adapted from code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        model = read_atomic_model_from_cif(filename)
        return cls.from_gemmi(
            model, n_voxels_per_side, voxel_size, mask, indexing, **kwargs
        )

    @classmethod
    def from_gemmi(
        cls: Type["VoxelCloud"],
        model,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        mask: bool = True,
        indexing: str = "xy",
        **kwargs: Any,
    ):
        coords, a_vals, b_vals = get_scattering_info_from_gemmi_model(model)

        coordinates_3d = make_coordinates(n_voxels_per_side, voxel_size)
        density = _build_real_space_voxels_from_atoms(
            coords, a_vals, b_vals, coordinates_3d
        )

        flat_density, flat_coordinates = flatten_and_coordinatize(
            density, voxel_size, mask, indexing
        )

        vdict = {
            "weights": flat_density,
            "coordinates": flat_coordinates,
            "voxel_size": voxel_size,
        }

        return cls(**vdict, **kwargs)


def _eval_3d_real_space_gaussian(
    coordinate_system: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    a: float,
    b: float,
) -> Float[Array, "N1 N2 N3"]:
    """
    Evaluate a gaussian on a 3D grid.
    The naming convention for parameters follows ``Robust
    Parameterization of Elastic and Absorptive Electron Atomic Scattering
    Factors'' by Peng et al.

    Parameters
    ----------
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinate_system of the grid.
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
    b_inverse = 4.0 * jnp.pi / b
    sq_distances = jnp.sum(
        b_inverse * (coordinate_system - atom_position) ** 2, axis=-1
    )
    density = jnp.exp(-jnp.pi * sq_distances) * a * b_inverse ** (3.0 / 2.0)
    return density


def _eval_3d_atom_potential(
    coordinate_system: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    atomic_as: Float[Array, "5"],
    atomic_bs: Float[Array, "5"],
) -> Float[Array, "N1 N2 N3"]:
    """
    Evaluates the electron potential of a single atom on a 3D grid.

    Parameters
    ----------
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinate_system of the grid.
    atom_position : `Array`, shape `(3,)`
        The location of the atom.
    atomic_as : `Array`, shape `(5,)`
        The intensity values for each gaussian in the atom.
    atomic_bs : `Array`, shape `(5,)`
        The inverse scale factors for each gaussian in the atom.

    Returns
    -------
    potential : `Array`, shape `(N1, N2, N3)`
        The potential of the atom evaluate on the grid.
    """
    eval_fxn = jax.vmap(
        _eval_3d_real_space_gaussian, in_axes=(None, None, 0, 0)
    )
    return jnp.sum(
        eval_fxn(coordinate_system, atom_position, atomic_as, atomic_bs),
        axis=0,
    )


@jax.jit
def _build_real_space_voxels_from_atoms(
    atom_positions: Float[Array, "N 3"],
    ff_a: Float[Array, "N 5"],
    ff_b: Float[Array, "N 5"],
    coordinate_system: Float[Array, "N1 N2 N3 3"],
) -> Tuple[_CubicVolume, _VolumeSliceCoords]:
    """
    Build a voxel representation of an atomic model.

    Parameters
    ----------
    atom_coords : `Array`, shape `(N, 3)`
        The coordinates of the atoms.
    ff_a : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        Intensity values for each Gaussian in the atom
    ff_b : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        The inverse scale factors for each Gaussian in the atom
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinates of each voxel in the grid.

    Returns
    -------
    density :  `Array`, shape `(N1, N2, N3)`
        The voxel representation of the atomic model.
    z_plane_coordinates : `Array`, shape `(N1, N2, 3)`
        The coordinates of each voxel in the z=0 plane.
    """
    density = jnp.zeros(coordinate_system.shape[:-1])

    def add_gaussian_to_density(i, density):
        density += _eval_3d_atom_potential(
            coordinate_system, atom_positions[i], ff_a[i], ff_b[i]
        )
        return density

    density = jax.lax.fori_loop(
        0, atom_positions.shape[0], add_gaussian_to_density, density
    )

    return density

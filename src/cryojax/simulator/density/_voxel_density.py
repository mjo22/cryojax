"""
Voxel-based electron density representations.
"""

__all__ = [
    "Voxels",
    "VoxelT",
    "RealVoxelGrid",
    "FourierVoxelGrid",
    "FourierVoxelGridAsSpline",
    "VoxelCloud",
]

import pathlib
from abc import abstractmethod
from typing import (
    Any,
    Tuple,
    Type,
    ClassVar,
    TypeVar,
    Optional,
    overload,
)
from typing_extensions import Self
from jaxtyping import Float, Array, Int
from equinox import AbstractVar
from functools import cached_property

import equinox as eqx
import jax
import jax.numpy as jnp

from ._electron_density import ElectronDensity
from ..pose import Pose
from ...io import (
    load_atoms,
    load_mrc,
    get_atom_info_from_gemmi_model,
    read_atoms_from_pdb,
    read_atoms_from_cif,
    mdtraj_load_from_file,
)
from ...core import field
from ...image import (
    pad,
    crop,
    fftn,
    make_coordinates,
    compute_spline_coefficients,
    CoordinateGrid,
    CoordinateList,
    FrequencySlice,
)
from ...typing import (
    RealCloud,
    RealVolume,
    RealCubicVolume,
    ComplexCubicVolume,
    VolumeCoords,
    VolumeSliceCoords,
    Real_,
)

VoxelT = TypeVar("VoxelT", bound="Voxels")
"""Type hint for a voxel-based electron density."""


class Voxels(ElectronDensity):
    """
    Voxel-based electron density representation.

    Attributes
    ----------
    weights :
        The electron density.
    voxel_size
        The voxel size of the electron density.
    """

    voxel_size: Real_ = field()

    @overload
    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelT],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: None,
        **kwargs: Any,
    ) -> VoxelT:
        ...

    @overload
    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelT],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: VolumeCoords,
        **kwargs: Any,
    ) -> VoxelT:
        ...

    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelT],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        **kwargs: Any,
    ) -> VoxelT:
        """
        Load a Voxels object from real-valued 3D electron
        density map.
        """
        raise NotImplementedError

    @classmethod
    def from_atoms(
        cls: Type[VoxelT],
        atom_positions: Float[Array, "N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: float,
        coordinate_grid_in_angstroms: VolumeCoords,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> VoxelT:
        """
        Load a Voxels object from atom positions and identities.
        """
        a_vals, b_vals = load_atoms.get_form_factor_params(
            atom_identities, form_factors
        )

        density = _build_real_space_voxels_from_atoms(
            atom_positions, a_vals, b_vals, coordinate_grid_in_angstroms
        )

        return cls.from_density_grid(
            density,
            voxel_size,
            coordinate_grid_in_angstroms / voxel_size,
            **kwargs,
        )

    @classmethod
    def from_trajectory(
        cls: Type[VoxelT],
        trajectory: Float[Array, "M N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: float,
        coordinate_grid_in_angstroms: VolumeCoords,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> VoxelT:
        a_vals, b_vals = load_atoms.get_form_factor_params(
            atom_identities, form_factors
        )

        _build_real_space_voxels_from_atomic_trajectory = jax.vmap(
            _build_real_space_voxels_from_atoms, (0, None, None, None), 0
        )

        _build_real_space_voxels_from_atoms(
            trajectory[0], a_vals, b_vals, coordinate_grid_in_angstroms
        )

        density = _build_real_space_voxels_from_atomic_trajectory(
            trajectory, a_vals, b_vals, coordinate_grid_in_angstroms
        )

        from_density_grid_vmap = jax.vmap(
            lambda d, vs, c: cls.from_density_grid(d, vs, c, **kwargs),
            in_axes=[0, 0, None],
        )
        return from_density_grid_vmap(
            density,
            jnp.full(density.shape[0], voxel_size),
            coordinate_grid_in_angstroms / voxel_size,
        )

    @classmethod
    def from_gemmi(
        cls: Type[VoxelT],
        model,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: Real_ | float = 1.0,
        **kwargs: Any,
    ) -> VoxelT:
        """
        Loads a PDB file as a Voxels subclass.  Uses the Gemmi library.
        Heavily based on a code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        atom_positions, atom_elements = get_atom_info_from_gemmi_model(model)

        coordinate_grid_in_angstroms = make_coordinates(
            n_voxels_per_side, voxel_size
        )

        return cls.from_atoms(
            atom_positions,
            atom_elements,
            voxel_size,
            coordinate_grid_in_angstroms,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls: Type[VoxelT],
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> VoxelT:
        """Load a voxel-based electron density."""
        path = pathlib.Path(filename)
        if path.suffix == ".mrc":
            return cls.from_mrc(filename, *args, **kwargs)
        elif path.suffix == ".pdb":
            return cls.from_pdb(filename, *args, **kwargs)
        elif path.suffix == ".cif":
            return cls.from_cif(filename, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"File format {path.suffix} not supported."
            )

    @classmethod
    def from_mrc(
        cls: Type[VoxelT],
        filename: str,
        **kwargs: Any,
    ) -> VoxelT:
        """Load Voxels from MRC file format."""
        density_grid, voxel_size = load_mrc(filename)
        return cls.from_density_grid(
            jnp.asarray(density_grid), voxel_size, **kwargs
        )

    @classmethod
    def from_pdb(
        cls: Type[VoxelT],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: Real_ | float = 1.0,
        **kwargs: Any,
    ) -> VoxelT:
        """Load Voxels from PDB file format."""
        atom_positions, atom_elements = read_atoms_from_pdb(filename)
        coordinate_grid_in_angstroms = make_coordinates(
            n_voxels_per_side, voxel_size
        )

        return cls.from_atoms(
            atom_positions,
            atom_elements,
            voxel_size,
            coordinate_grid_in_angstroms,
            **kwargs,
        )

    @classmethod
    def from_cif(
        cls: Type[VoxelT],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: Real_ | float = 1.0,
        **kwargs: Any,
    ) -> VoxelT:
        """Load Voxels from CIF file format."""
        atom_positions, atom_elements = read_atoms_from_cif(filename)
        coordinate_grid_in_angstroms = make_coordinates(
            n_voxels_per_side, voxel_size
        )

        return cls.from_atoms(
            atom_positions,
            atom_elements,
            voxel_size,
            coordinate_grid_in_angstroms,
            **kwargs,
        )

    @classmethod
    def from_mdtraj(
        cls: Type[VoxelT],
        trajectory_path: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: Real_ | float = 1.0,
        topology_file: str = None,
        **kwargs: Any,
    ) -> VoxelT:
        """
        Load Voxels from MDTraj trajectory.

        Parameters
        ----------
        trajectory_path : str
            Path to trajectory file.
        n_voxels_per_side : tuple of int
            Number of voxels per side.
        voxel_size : float
            Size of each voxel in angstroms.
        topology_file : str, optional
            Path to topology file, if required to load the trajectory.

        Returns
        -------
        VoxelT
            A subclass of Voxels.

        Notes
        -----
        Returns a Voxel object with
        a nontrivial indexed dimension (the first dimension): scattering or
        otherwise using the density may require vmaps!
        """
        trajectory, atom_identities = mdtraj_load_from_file(
            trajectory_path, topology_file
        )
        coordinate_grid_in_angstroms = make_coordinates(
            n_voxels_per_side, voxel_size
        )
        return cls.from_trajectory(
            trajectory,
            atom_identities,
            voxel_size,
            coordinate_grid_in_angstroms,
            None,
        )


class FourierVoxelGrid(Voxels):
    """
    Abstraction of a 3D electron density voxel grid
    in fourier space.

    Attributes
    ----------
    weights :
        3D electron density grid in fourier space.
    frequency_slice :
        Central slice of cartesian coordinate system
        in fourier space.
    """

    fourier_density_grid: ComplexCubicVolume = field()
    frequency_slice: FrequencySlice = field()

    is_real: ClassVar[bool] = False

    def __init__(
        self,
        fourier_density_grid: ComplexCubicVolume,
        frequency_slice: FrequencySlice,
        voxel_size: Real_,
    ):
        self.fourier_density_grid = fourier_density_grid
        self.frequency_slice = frequency_slice
        self.voxel_size = voxel_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fourier_density_grid.shape

    @cached_property
    def frequency_slice_in_angstroms(self) -> FrequencySlice:
        return self.frequency_slice / self.voxel_size

    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        return eqx.tree_at(
            lambda d: d.frequency_slice,
            self,
            FrequencySlice(
                pose.rotate(self.frequency_slice.get(), is_real=self.is_real)
            ),
            is_leaf=lambda x: isinstance(x, FrequencySlice),
        )

    @classmethod
    def from_density_grid(
        cls: Type["FourierVoxelGrid"],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
    ) -> "FourierVoxelGrid":
        # Change how template sits in box to match cisTEM
        density_grid = jnp.transpose(density_grid, axes=[-1, -2, -3])
        # Pad template
        if pad_scale < 1.0:
            raise ValueError("pad_scale must be greater than 1.0")
        # ... always pad to even size to avoid interpolation issues in
        # fourier slice extraction.
        padded_shape = tuple(
            [int(s * pad_scale) for s in density_grid.shape[-3:]]
        )
        padded_density_grid = pad(density_grid, padded_shape, mode=pad_mode)
        # Load density and coordinates. For now, do not store the
        # fourier density only on the half space. Fourier slice extraction
        # does not currently work if rfftn is us
        # ... store the density grid with the zero frequency component in the center
        fourier_density_grid = jnp.fft.fftshift(
            fftn(padded_density_grid, axes=(-3, -2, -1))
        )
        # ... create in-plane frequency slice on the half space
        frequency_slice = FrequencySlice(
            shape=padded_density_grid.shape[-3:-1], half_space=False
        )

        return cls(fourier_density_grid, frequency_slice, voxel_size)


class FourierVoxelGridAsSpline(FourierVoxelGrid):
    """
    Abstraction of a 3D electron density voxel grid
    in fourier space.

    Attributes
    ----------
    weights :
        3D electron density grid in fourier space.
    frequency_slice :
        Central slice of cartesian coordinate system
        in fourier space.
    """

    spline_coefficients: ComplexCubicVolume = field()
    frequency_slice: FrequencySlice = field()
    fourier_density_grid: None = field()

    is_real: ClassVar[bool] = False

    def __init__(
        self,
        fourier_density_grid: ComplexCubicVolume,
        frequency_slice: FrequencySlice,
        voxel_size: Real_,
    ):
        self.spline_coefficients = compute_spline_coefficients(
            fourier_density_grid
        )
        self.frequency_slice = frequency_slice
        self.voxel_size = voxel_size
        self.fourier_density_grid = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple([s - 2 for s in self.spline_coefficients.shape])


class RealVoxelGrid(Voxels):
    """
    Abstraction of a 3D electron density voxel grid.
    The voxel grid is given in real space.

    Attributes
    ----------
    weights :
        3D electron density voxel grid.
    coordinate_list :
        List of coordinates for the point cloud.
    """

    density_grid: RealVolume = field()
    coordinate_grid: CoordinateGrid = field()

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        density_grid: ComplexCubicVolume,
        coordinate_grid: CoordinateGrid,
        voxel_size: Real_,
    ):
        self.density_grid = density_grid
        self.coordinate_grid = coordinate_grid
        self.voxel_size = voxel_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.density_grid.shape

    @cached_property
    def coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.voxel_size * self.coordinate_grid

    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_grid,
            self,
            CoordinateGrid(
                pose.rotate(self.coordinate_grid.get(), is_real=self.is_real)
            ),
            is_leaf=lambda x: isinstance(x, CoordinateGrid),
        )

    @overload
    @classmethod
    def from_density_grid(
        cls: Type["RealVoxelGrid"],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: VolumeCoords,
        crop_scale: None,
    ) -> "RealVoxelGrid":
        ...

    @overload
    @classmethod
    def from_density_grid(
        cls: Type["RealVoxelGrid"],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: None,
        crop_scale: Optional[float],
    ) -> "RealVoxelGrid":
        ...

    @classmethod
    def from_density_grid(
        cls: Type["RealVoxelGrid"],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        crop_scale: Optional[float] = None,
    ) -> "RealVoxelGrid":
        # Change how template sits in the box.
        # Ideally we would change this in the same way for all
        # I/O methods. However, the algorithms used all
        # have their own xyz conventions. The choice here is to
        # make jax-finufft output match cisTEM.
        density_grid = jnp.transpose(density_grid, axes=[-2, -1, -3])
        # Make coordinates if not given
        if coordinate_grid is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale > 1.0:
                    raise ValueError("crop_scale must be less than 1.0")
                cropped_shape = tuple(
                    [int(s * crop_scale) for s in density_grid.shape[-3:]]
                )
                density_grid = crop(density_grid, cropped_shape)
            coordinate_grid = CoordinateGrid(shape=density_grid.shape[-3:])

        return cls(density_grid, coordinate_grid, voxel_size)


class VoxelCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    The point cloud is given in real space.

    Attributes
    ----------
    weights :
        Flattened 3D electron density voxel grid into a
        point cloud.
    coordinate_list :
        List of coordinates for the point cloud.
    """

    density_weights: RealCloud = field()
    coordinate_list: CoordinateList = field()

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        density_weights: ComplexCubicVolume,
        coordinate_list: CoordinateList,
        voxel_size: Real_,
    ):
        self.density_weights = density_weights
        self.coordinate_list = coordinate_list
        self.voxel_size = voxel_size

    @property
    def shape(self) -> tuple[int, int]:
        return self.density_weights.shape

    @cached_property
    def coordinate_list_in_angstroms(self) -> CoordinateList:
        return self.voxel_size * self.coordinate_list

    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_list,
            self,
            CoordinateList(
                pose.rotate(self.coordinate_list.get(), is_real=self.is_real)
            ),
            is_leaf=lambda x: isinstance(x, CoordinateList),
        )

    @classmethod
    def from_density_grid(
        cls: Type["VoxelCloud"],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
    ) -> "VoxelCloud":
        # Change how template sits in the box.
        # Ideally we would change this in the same way for all
        # I/O methods. However, the algorithms used all
        # have their own xyz conventions. The choice here is to
        # make jax-finufft output match cisTEM.
        density_grid = jnp.transpose(density_grid, axes=[-2, -1, -3])
        # Make coordinates if not given
        if coordinate_grid is None:
            coordinate_grid = make_coordinates(density_grid.shape)
        # ... mask zeros to store smaller arrays. This
        # option is not jittable.
        nonzero = jnp.where(~jnp.isclose(density_grid, 0.0))
        flat_density = density_grid[nonzero]
        coordinate_list = coordinate_grid[nonzero]

        return cls(flat_density, CoordinateList(coordinate_list), voxel_size)


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
) -> Tuple[RealCubicVolume, VolumeSliceCoords]:
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

from abc import abstractmethod
from typing import (
    Any,
    Type,
    ClassVar,
    Optional,
    overload,
)
from typing_extensions import Self, override
from jaxtyping import Float, Array, Int
from functools import cached_property
from equinox import field, AbstractVar, AbstractClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ._electron_density import AbstractElectronDensity
from .._pose import AbstractPose
from ...io import get_form_factor_params

from ...image.operators import AbstractFilter
from ...image import (
    pad_to_shape,
    crop_to_shape,
    fftn,
    compute_spline_coefficients,
)
from ...coordinates import CoordinateGrid, CoordinateList, FrequencySlice
from ...typing import (
    RealCloud,
    RealVolume,
    RealCubicVolume,
    ComplexCubicVolume,
    VolumeSliceCoords,
    Real_,
)


class AbstractVoxels(AbstractElectronDensity, strict=True):
    """Abstract interface for a voxel-based electron density representation."""

    voxel_size: AbstractVar[Real_]
    """The voxel size of the electron density."""
    is_real: AbstractClassVar[bool]
    """Whether or not the representation is real or fourier-space."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of electron density voxel array."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        **kwargs: Any,
    ) -> Self:
        """Load an `AbstractVoxels` from real-valued 3D electron
        density map.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_atoms(
        cls: Type[Self],
        atom_positions: Float[Array, "N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: Real_ | float,
        coordinate_grid_in_angstroms: CoordinateGrid,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> Self:
        """Load an `AbstractVoxels` from atom positions and identities."""
        raise NotImplementedError


class AbstractFourierVoxelGrid(AbstractVoxels, strict=True):
    """Abstract interface of a 3D electron density voxel grid
    in fourier-space.
    """

    frequency_slice: AbstractVar[FrequencySlice]

    @abstractmethod
    def __init__(
        self,
        fourier_density_grid: ComplexCubicVolume,
        frequency_slice: FrequencySlice,
        voxel_size: Real_,
    ):
        raise NotImplementedError

    @cached_property
    def frequency_slice_in_angstroms(self) -> FrequencySlice:
        """The `frequency_slice` in angstroms."""
        return self.frequency_slice / self.voxel_size

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        return eqx.tree_at(
            lambda d: d.frequency_slice.array,
            self,
            pose.rotate_coordinates(self.frequency_slice.get(), inverse=True),
        )

    @classmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: Optional[AbstractFilter] = None,
    ) -> Self:
        """Load an `AbstractFourierVoxelGrid` from real-valued 3D electron
        density map.

        **Arguments:**

        `density_grid`: An electron density voxel grid in real space.

        `voxel_size`: The voxel size of `density_grid`.

        `pad_scale`: Scale factor at which to pad `density_grid` before fourier
                     transform. Must be a value greater than `1.0`.

        `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.

        `filter`: A filter to apply to the result of the fourier transform of
                  `density_grid`, i.e. `fftn(density_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Pad template
        if pad_scale < 1.0:
            raise ValueError("pad_scale must be greater than 1.0")
        # ... always pad to even size to avoid interpolation issues in
        # fourier slice extraction.
        padded_shape = tuple([int(s * pad_scale) for s in density_grid.shape])
        padded_density_grid = pad_to_shape(density_grid, padded_shape, mode=pad_mode)
        # Load density and coordinates. For now, do not store the
        # fourier density only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_density_grid_with_zero_in_corner = (
            fftn(padded_density_grid)
            if filter is None
            else filter(fftn(padded_density_grid))
        )
        # ... store the density grid with the zero frequency component in the center
        fourier_density_grid = jnp.fft.fftshift(
            fourier_density_grid_with_zero_in_corner
        )
        # ... create in-plane frequency slice on the half space
        frequency_slice = FrequencySlice(
            padded_density_grid.shape[:-1], half_space=False
        )

        return cls(fourier_density_grid, frequency_slice, jnp.asarray(voxel_size))

    @classmethod
    def from_atoms(
        cls: Type[Self],
        atom_positions: Float[Array, "N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: Real_ | float,
        coordinate_grid_in_angstroms: CoordinateGrid,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> Self:
        """Load an `AbstractFourierVoxelGrid` from atom positions and identities.

        **Arguments:**

        - `**kwargs`: Passed to `AbstractFourierVoxelGrid.from_density_grid`
        """
        a_vals, b_vals = get_form_factor_params(atom_identities, form_factors)

        density = build_real_space_voxels_from_atoms(
            atom_positions, a_vals, b_vals, coordinate_grid_in_angstroms.get()
        )

        return cls.from_density_grid(
            density,
            voxel_size,
            **kwargs,
        )


class FourierVoxelGrid(AbstractFourierVoxelGrid):
    """A 3D electron density voxel grid in fourier-space."""

    fourier_density_grid: ComplexCubicVolume = field(converter=jnp.asarray)
    """The voxel grid in fourier space."""
    frequency_slice: FrequencySlice
    """Frequency slice coordinate system."""
    voxel_size: Real_ = field(converter=jnp.asarray)
    """The voxel size."""

    is_real: ClassVar[bool] = False

    @override
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


class FourierVoxelGridInterpolator(AbstractFourierVoxelGrid):
    """A 3D electron density voxel grid in fourier-space, represented
    by spline coefficients.
    """

    coefficients: ComplexCubicVolume = field(converter=jnp.asarray)
    """Cubic spline coefficients for the voxel grid."""
    frequency_slice: FrequencySlice
    """Frequency slice coordinate system."""
    voxel_size: Real_ = field(converter=jnp.asarray)
    """The voxel size."""

    is_real: ClassVar[bool] = False

    def __init__(
        self,
        fourier_density_grid: ComplexCubicVolume,
        frequency_slice: FrequencySlice,
        voxel_size: Real_,
    ):
        """
        !!! note
            The argument `fourier_density_grid` is used to set
            `FourierVoxelGridInterpolator.coefficients` in the `__init__`.
            For example,

            ```python
            voxels = FourierVoxelGridInterpolator(fourier_density_grid, frequency_slice, voxel_size)
            assert not hasattr(voxels, "fourier_density_grid")  # This does not store the `fourier_voxel_grid`
            assert hasattr(voxels, "coefficients")  # Instead it computes `coefficients` upon `__init__`
            ```
        """
        self.coefficients = compute_spline_coefficients(fourier_density_grid)
        self.frequency_slice = frequency_slice
        self.voxel_size = voxel_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple([s - 2 for s in self.coefficients.shape])


class RealVoxelGrid(AbstractVoxels, strict=True):
    """Abstraction of a 3D electron density voxel grid.
    The voxel grid is given in real-space.
    """

    density_grid: RealCubicVolume = field(converter=jnp.asarray)
    """A cubic voxel grid in real-space."""
    coordinate_grid: CoordinateGrid
    """A coordinate grid."""
    voxel_size: Real_ = field(converter=jnp.asarray)
    """The voxel size."""

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        density_grid: RealCubicVolume,
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
        """The `coordinate_grid` in angstroms."""
        return self.voxel_size * self.coordinate_grid

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        return eqx.tree_at(
            lambda d: d.coordinate_grid.array,
            self,
            pose.rotate_coordinates(self.coordinate_grid.get(), inverse=False),
        )

    @overload
    @classmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: CoordinateGrid,
        *,
        crop_scale: None,
    ) -> Self: ...

    @overload
    @classmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float,
        coordinate_grid: None,
        *,
        crop_scale: Optional[float],
    ) -> Self: ...

    @classmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[CoordinateGrid] = None,
        *,
        crop_scale: Optional[float] = None,
    ) -> Self:
        """Load an `RealVoxelGrid` from real-valued 3D electron
        density map.

        !!! warning
            `density_grid` is transposed upon instantiation in order to make
            the results of `cryojax.simulator.NufftProject` agree with
            `cryojax.simulator.FourierSliceExtract`.

            ```python
            density_grid = ...
            voxels = RealVoxelGrid.from_density_grid(density_grid, ...)
            assert density_grid == jnp.transpose(voxels.density_grid, axes=[1, 0, 2])
            ```

        **Arguments:**

        `density_grid`: An electron density voxel grid in real space.

        `voxel_size`: The voxel size of `density_grid`.

        `crop_scale`: Scale factor at which to crop `density_grid`.
                      Must be a value less than `1.0`.
        """
        # A nasty hack to make NufftProject agree with FourierSliceExtract
        density_grid = jnp.transpose(density_grid, axes=[1, 0, 2])
        # Make coordinates if not given
        if coordinate_grid is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale > 1.0:
                    raise ValueError("crop_scale must be less than 1.0")
                cropped_shape = tuple(
                    [int(s * crop_scale) for s in density_grid.shape[-3:]]
                )
                density_grid = crop_to_shape(density_grid, cropped_shape)
            coordinate_grid = CoordinateGrid(density_grid.shape[-3:])

        return cls(density_grid, coordinate_grid, jnp.asarray(voxel_size))

    @classmethod
    def from_atoms(
        cls: Type[Self],
        atom_positions: Float[Array, "N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: Real_ | float,
        coordinate_grid_in_angstroms: CoordinateGrid,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> Self:
        """Load a `RealVoxelGrid` from atom positions and identities.

        **Arguments:**

        - `**kwargs`: Passed to `RealVoxelGrid.from_density_grid`
        """
        a_vals, b_vals = get_form_factor_params(atom_identities, form_factors)

        density = build_real_space_voxels_from_atoms(
            atom_positions, a_vals, b_vals, coordinate_grid_in_angstroms.get()
        )

        return cls.from_density_grid(
            density,
            voxel_size,
            coordinate_grid_in_angstroms / voxel_size,
            **kwargs,
        )


class RealVoxelCloud(AbstractVoxels, strict=True):
    """Abstraction of a 3D electron density voxel point cloud.

    !!! info
        This object is similar to the `RealVoxelGrid`. Instead
        of storing the whole voxel grid, a `RealVoxelCloud` need
        only store points of non-zero electron density. Therefore,
        a `RealVoxelCloud` stores a point cloud of electron density
        voxel values.
    """

    density_weights: RealCloud = field(converter=jnp.asarray)
    """A point-cloud of voxel density values."""
    coordinate_list: CoordinateList
    """Coordinate list for the `density_weights`."""
    voxel_size: Real_ = field(converter=jnp.asarray)
    """The voxel size."""

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        density_weights: RealCloud,
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
        """The `coordinate_list` in angstroms."""
        return self.voxel_size * self.coordinate_list

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        return eqx.tree_at(
            lambda d: d.coordinate_list.array,
            self,
            pose.rotate_coordinates(self.coordinate_list.get(), inverse=False),
        )

    @classmethod
    def from_density_grid(
        cls: Type[Self],
        density_grid: RealVolume,
        voxel_size: Real_ | float = 1.0,
        coordinate_grid: Optional[CoordinateGrid] = None,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> Self:
        """Load an `RealVoxelCloud` from real-valued 3D electron
        density map.

        !!! warning
            `density_grid` is transposed upon instantiation in order to make
            the results of [`cryojax.simulator.NufftProject`][] agree with
            [`cryojax.simulator.FourierSliceExtract`][].
            See [`cryojax.simulator.RealVoxelGrid`][] for more detail.

        **Arguments:**

        `density_grid`: An electron density voxel grid in real space.

        `voxel_size`: The voxel size of `density_grid`.

        `rtol`: Argument passed to `jnp.isclose`, used for removing
                points of zero electron density.

        `atol`: Argument passed to `jnp.isclose`, used for removing
                points of zero electron density.
        """
        # A nasty hack to make NufftProject agree with FourierSliceExtract
        density_grid = jnp.transpose(density_grid, axes=[1, 0, 2])
        # Make coordinates if not given
        if coordinate_grid is None:
            coordinate_grid = CoordinateGrid(density_grid.shape)
        # ... mask zeros to store smaller arrays. This
        # option is not jittable.
        nonzero = jnp.where(~jnp.isclose(density_grid, 0.0, rtol=rtol, atol=atol))
        flat_density = density_grid[nonzero]
        coordinate_list = CoordinateList(coordinate_grid.get()[nonzero])

        return cls(flat_density, coordinate_list, jnp.asarray(voxel_size))

    @classmethod
    def from_atoms(
        cls: Type[Self],
        atom_positions: Float[Array, "N 3"],
        atom_identities: Int[Array, "N"],
        voxel_size: Real_ | float,
        coordinate_grid_in_angstroms: CoordinateGrid,
        form_factors: Optional[Float[Array, "N 5"]] = None,
        **kwargs: Any,
    ) -> Self:
        """Load a `RealVoxelCloud` from atom positions and identities.

        **Arguments:**

        - `**kwargs`: Passed to `RealVoxelCloud.from_density_grid`
        """
        a_vals, b_vals = get_form_factor_params(atom_identities, form_factors)

        density = build_real_space_voxels_from_atoms(
            atom_positions, a_vals, b_vals, coordinate_grid_in_angstroms.get()
        )

        return cls.from_density_grid(
            density,
            voxel_size,
            coordinate_grid_in_angstroms / voxel_size,
            **kwargs,
        )


def evaluate_3d_real_space_gaussian(
    coordinate_grid_in_angstroms: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    a: float,
    b: float,
) -> Float[Array, "N1 N2 N3"]:
    """Evaluate a gaussian on a 3D grid.
    The naming convention for parameters follows "Robust
    Parameterization of Elastic and Absorptive Electron Atomic Scattering
    Factors" by Peng et al.

    **Arguments:**

    `coordinate_grid`: The coordinate system of the grid.

    `pos`: The center of the gaussian.

    `a`: A scale factor.

    `b`: The scale of the gaussian.

    **Returns:**

    `density`: The density of the gaussian on the grid.
    """
    b_inverse = 4.0 * jnp.pi / b
    sq_distances = jnp.sum(
        b_inverse * (coordinate_grid_in_angstroms - atom_position) ** 2, axis=-1
    )
    density = jnp.exp(-jnp.pi * sq_distances) * a * b_inverse ** (3.0 / 2.0)
    return density


def evaluate_3d_atom_potential(
    coordinate_grid_in_angstroms: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    atomic_as: Float[Array, "5"],
    atomic_bs: Float[Array, "5"],
) -> Float[Array, "N1 N2 N3"]:
    """Evaluates the electron potential of a single atom on a 3D grid.

    **Arguments:**

    `coordinate_grid_in_angstroms`: The coordinate system of the grid.

    `atom_position`: The location of the atom.

    `atomic_as`: The intensity values for each gaussian in the atom.

    `atomic_bs`: The inverse scale factors for each gaussian in the atom.

    **Returns:**

    `potential`: The potential of the atom evaluate on the grid.
    """
    eval_fxn = jax.vmap(evaluate_3d_real_space_gaussian, in_axes=(None, None, 0, 0))
    return jnp.sum(
        eval_fxn(coordinate_grid_in_angstroms, atom_position, atomic_as, atomic_bs),
        axis=0,
    )


@jax.jit
def build_real_space_voxels_from_atoms(
    atom_positions: Float[Array, "N 3"],
    ff_a: Float[Array, "N 5"],
    ff_b: Float[Array, "N 5"],
    coordinate_grid_in_angstroms: Float[Array, "N1 N2 N3 3"],
) -> tuple[RealCubicVolume, VolumeSliceCoords]:
    """
    Build a voxel representation of an atomic model.

    **Arguments**

    `atom_coords`: The coordinates of the atoms.

    `ff_a`: Intensity values for each Gaussian in the atom

    `ff_b` : The inverse scale factors for each Gaussian in the atom

    `coordinate_grid` : The coordinates of each voxel in the grid.

    **Returns:**

    `density`: The voxel representation of the atomic model.
    """
    density = jnp.zeros(coordinate_grid_in_angstroms.shape[:-1])

    def add_gaussian_to_density(i, density):
        density += evaluate_3d_atom_potential(
            coordinate_grid_in_angstroms, atom_positions[i], ff_a[i], ff_b[i]
        )
        return density

    density = jax.lax.fori_loop(
        0, atom_positions.shape[0], add_gaussian_to_density, density
    )

    return density

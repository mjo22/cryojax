"""
Voxel-based representations of the scattering potential.
"""

from abc import abstractmethod
from functools import cached_property
from typing import cast, ClassVar, Optional
from typing_extensions import override, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox import AbstractClassVar, AbstractVar, field
from jaxtyping import Array, Complex, Float

from ..._errors import error_if_not_positive
from ...coordinates import CoordinateGrid, CoordinateList, FrequencySlice
from ...image import (
    compute_spline_coefficients,
    crop_to_shape,
    fftn,
    pad_to_shape,
)
from ...image.operators import AbstractFilter
from .._pose import AbstractPose
from .base_potential import AbstractPotentialRepresentation


class AbstractVoxelPotential(AbstractPotentialRepresentation, strict=True):
    """Abstract interface for a voxel-based scattering potential representation."""

    voxel_size: AbstractVar[Float[Array, ""]]
    is_real: AbstractClassVar[bool]

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the voxel array."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    ) -> Self:
        """Load an `AbstractVoxelPotential` from real-valued 3D electron
        scattering potential.
        """
        raise NotImplementedError


# Not public API
class AbstractFourierVoxelGridPotential(AbstractVoxelPotential, strict=True):
    """Abstract interface of a 3D scattering potential voxel grid
    in fourier-space.
    """

    wrapped_frequency_slice_in_pixels: AbstractVar[FrequencySlice]

    @abstractmethod
    def __init__(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        wrapped_frequency_slice_in_pixels: FrequencySlice,
        voxel_size: Float[Array, ""] | float,
    ):
        raise NotImplementedError

    @cached_property
    def wrapped_frequency_slice_in_angstroms(self) -> FrequencySlice:
        """The `wrapped_frequency_slice_in_pixels` in angstroms."""
        return self.wrapped_frequency_slice_in_pixels / self.voxel_size

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated
        `wrapped_frequency_slice_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.wrapped_frequency_slice_in_pixels.array,
            self,
            pose.rotate_coordinates(
                self.wrapped_frequency_slice_in_pixels.get(), inverse=True
            ),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: Optional[AbstractFilter] = None,
    ) -> Self:
        """Load an `AbstractFourierVoxelGridPotential` from real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `pad_scale`: Scale factor at which to pad `real_voxel_grid` before fourier
                     transform. Must be a value greater than `1.0`.
        - `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.
        - `filter`: A filter to apply to the result of the fourier transform of
                  `real_voxel_grid`, i.e. `fftn(real_voxel_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
        # Pad template
        if pad_scale < 1.0:
            raise ValueError("`pad_scale` must be greater than 1.0")
        # ... always pad to even size to avoid interpolation issues in
        # fourier slice extraction.
        padded_shape = cast(
            tuple[int, int, int],
            tuple([int(s * pad_scale) for s in real_voxel_grid.shape]),
        )
        padded_real_voxel_grid = pad_to_shape(
            real_voxel_grid, padded_shape, mode=pad_mode
        )
        # Load potential and coordinates. For now, do not store the
        # fourier potential only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_voxel_grid_with_zero_in_corner = (
            fftn(padded_real_voxel_grid)
            if filter is None
            else filter(fftn(padded_real_voxel_grid))
        )
        # ... store the potential grid with the zero frequency component in the center
        fourier_voxel_grid = jnp.fft.fftshift(fourier_voxel_grid_with_zero_in_corner)
        # ... create in-plane frequency slice on the half space
        frequency_slice = FrequencySlice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]), half_space=False
        )

        return cls(fourier_voxel_grid, frequency_slice, voxel_size)


class FourierVoxelGridPotential(AbstractFourierVoxelGridPotential):
    """A 3D scattering potential voxel grid in fourier-space."""

    fourier_voxel_grid: Complex[Array, "dim dim dim"]
    wrapped_frequency_slice_in_pixels: FrequencySlice
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real: ClassVar[bool] = False

    @override
    def __init__(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        wrapped_frequency_slice_in_pixels: FrequencySlice,
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `fourier_voxel_grid`: The cubic voxel grid in fourier space.
        - `wrapped_frequency_slice_in_pixels`: Frequency slice coordinate system,
                                               wrapped in a `FrequencySlice` object.
        - `voxel_size`: The voxel size.
        """
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid)
        self.wrapped_frequency_slice_in_pixels = wrapped_frequency_slice_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `fourier_voxel_grid`."""
        return cast(tuple[int, int, int], self.fourier_voxel_grid.shape)


class FourierVoxelGridPotentialInterpolator(AbstractFourierVoxelGridPotential):
    """A 3D scattering potential voxel grid in fourier-space, represented
    by spline coefficients.
    """

    coefficients: Float[Array, "coeff_dim coeff_dim coeff_dim"]
    wrapped_frequency_slice_in_pixels: FrequencySlice
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real: ClassVar[bool] = False

    def __init__(
        self,
        fourier_voxel_grid: Float[Array, "dim dim dim"],
        wrapped_frequency_slice_in_pixels: FrequencySlice,
        voxel_size: Float[Array, ""] | float,
    ):
        """
        !!! note
            The argument `fourier_voxel_grid` is used to set
            `FourierVoxelGridPotentialInterpolator.coefficients` in the `__init__`,
            but it is not stored in the class. For example,

            ```python
            voxels = FourierVoxelGridPotentialInterpolator(
                fourier_voxel_grid, frequency_slice, voxel_size
            )
            assert hasattr(voxels, "fourier_voxel_grid")  # This will return an error
            assert hasattr(voxels, "coefficients")  # Instead, store spline coefficients
            ```

        **Arguments:**

        - `fourier_voxel_grid`: The cubic voxel grid in fourier space.
        - `wrapped_frequency_slice_in_pixels`: Frequency slice coordinate system,
                                               wrapped in a `FrequencySlice` object.
        - `voxel_size`: The voxel size.
        """
        self.coefficients = compute_spline_coefficients(jnp.asarray(fourier_voxel_grid))
        self.wrapped_frequency_slice_in_pixels = wrapped_frequency_slice_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the original `fourier_voxel_grid` from which
        `coefficients` were computed.
        """
        return cast(tuple[int, int, int], tuple([s - 2 for s in self.coefficients.shape]))


class RealVoxelGridPotential(AbstractVoxelPotential, strict=True):
    """Abstraction of a 3D scattering potential voxel grid in real-space."""

    real_voxel_grid: Float[Array, "dim dim dim"]
    wrapped_coordinate_grid_in_pixels: CoordinateGrid
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        real_voxel_grid: Float[Array, "dim dim dim"],
        wrapped_coordinate_grid_in_pixels: CoordinateGrid,
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `real_voxel_grid`: The voxel grid in fourier space.
        - `wrapped_coordinate_grid_in_pixels`: A coordinate grid, wrapped into a
                                               `CoordinateGrid` object.
        - `voxel_size`: The voxel size.
        """
        self.real_voxel_grid = jnp.asarray(real_voxel_grid)
        self.wrapped_coordinate_grid_in_pixels = wrapped_coordinate_grid_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `real_voxel_grid`."""
        return cast(tuple[int, int, int], self.real_voxel_grid.shape)

    @cached_property
    def wrapped_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        """The `wrapped_coordinate_grid_in_pixels` in angstroms."""
        return self.voxel_size * self.wrapped_coordinate_grid_in_pixels  # type: ignore

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated
        `wrapped_coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.wrapped_coordinate_grid_in_pixels.array,
            self,
            pose.rotate_coordinates(
                self.wrapped_coordinate_grid_in_pixels.get(), inverse=False
            ),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        coordinate_grid_in_pixels: Optional[CoordinateGrid] = None,
        crop_scale: Optional[float] = None,
    ) -> Self:
        """Load a `RealVoxelGridPotential` from a real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: An electron scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
        # Make coordinates if not given
        if coordinate_grid_in_pixels is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale < 1.0:
                    raise ValueError("`crop_scale` must be greater than 1.0")
                cropped_shape = cast(
                    tuple[int, int, int],
                    tuple([int(s / crop_scale) for s in real_voxel_grid.shape[-3:]]),
                )
                real_voxel_grid = crop_to_shape(real_voxel_grid, cropped_shape)
            coordinate_grid_in_pixels = CoordinateGrid(real_voxel_grid.shape[-3:])

        return cls(real_voxel_grid, coordinate_grid_in_pixels, voxel_size)


class RealVoxelCloudPotential(AbstractVoxelPotential, strict=True):
    """Abstraction of a 3D electron scattering potential voxel point cloud.

    !!! info
        This object is similar to the `RealVoxelGridPotential`. Instead
        of storing the whole voxel grid, a `RealVoxelCloudPotential` need
        only store points of non-zero scattering potential. Therefore,
        a `RealVoxelCloudPotential` stores a point cloud of scattering potential
        voxel values. Instantiating with the `from_real_voxel_grid` constructor
        will automatically mask points of zero scattering potential.
    """

    voxel_weights: Float[Array, " size"]
    wrapped_coordinate_list_in_pixels: CoordinateList
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real: ClassVar[bool] = True

    def __init__(
        self,
        voxel_weights: Float[Array, " size"],
        wrapped_coordinate_list_in_pixels: CoordinateList,
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `voxel_weights`: A point-cloud of voxel scattering potential values.
        - `wrapped_coordinate_list_in_pixels`: Coordinate list for the `voxel_weights`,
                                               wrapped in a `CoordinateList` object.
        - `voxel_size`: The voxel size.
        """
        self.voxel_weights = jnp.asarray(voxel_weights)
        self.wrapped_coordinate_list_in_pixels = wrapped_coordinate_list_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int]:
        """The shape of `voxel_weights`."""
        return cast(tuple[int], self.voxel_weights.shape)

    @cached_property
    def wrapped_coordinate_list_in_angstroms(self) -> CoordinateList:
        """The `wrapped_coordinate_list_in_pixels` in angstroms."""
        return self.voxel_size * self.wrapped_coordinate_list_in_pixels  # type: ignore

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated
        `wrapped_coordinate_list_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.wrapped_coordinate_list_in_pixels.array,
            self,
            pose.rotate_coordinates(
                self.wrapped_coordinate_list_in_pixels.get(), inverse=False
            ),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        coordinate_grid_in_pixels: Optional[CoordinateGrid] = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        size: Optional[int] = None,
        fill_value: Optional[float] = None,
    ) -> Self:
        """Load an `RealVoxelCloudPotential` from a real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: An electron scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `rtol`: Argument passed to `jnp.isclose`, used for masking
                  voxels of zero scattering potential.
        - `atol`: Argument passed to `jnp.isclose`, used for masking
                  voxels of zero scattering potential.
        - `size`: Argument passed to `jnp.where`, used for fixing the size
                  of the masked scattering potential. This argument is required
                  for using this function with a JAX transformation.
        - `fill_value`: Argument passed to `jnp.where`, used if `size` is specified and
                        the mask has fewer than the indicated number of elements.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
        # Make coordinates if not given
        if coordinate_grid_in_pixels is None:
            coordinate_grid_in_pixels = CoordinateGrid(real_voxel_grid.shape)
        # ... mask zeros to store smaller arrays. This
        # option is not jittable.
        nonzero = jnp.where(
            ~jnp.isclose(real_voxel_grid, 0.0, rtol=rtol, atol=atol),
            size=size,
            fill_value=fill_value,
        )
        flat_potential = real_voxel_grid[nonzero]
        coordinate_list = CoordinateList(coordinate_grid_in_pixels.get()[nonzero])

        return cls(flat_potential, coordinate_list, voxel_size)
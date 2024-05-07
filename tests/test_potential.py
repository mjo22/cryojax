import jax.numpy as jnp
from jaxtyping import Array, Float

import cryojax.simulator as cs
from cryojax.coordinates import (
    CoordinateGrid,
    CoordinateList,
    FrequencySlice,
)


def test_voxel_electron_potential_loaders():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    voxel_size = 1.1
    fourier_potential = cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size=voxel_size
    )
    real_potential = cs.RealVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size=voxel_size
    )
    cloud_potential = cs.RealVoxelCloudPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size=voxel_size
    )
    for potential in [real_potential, fourier_potential, cloud_potential]:
        assert potential.voxel_size == jnp.asarray(voxel_size)

    assert isinstance(fourier_potential.wrapped_frequency_slice_in_pixels, FrequencySlice)
    assert isinstance(
        fourier_potential.wrapped_frequency_slice_in_pixels.get(),
        Float[Array, "1 _ _ 3"],
    )
    assert isinstance(real_potential.wrapped_coordinate_grid_in_pixels, CoordinateGrid)
    assert isinstance(
        real_potential.wrapped_coordinate_grid_in_pixels.get(), Float[Array, "_ _ _ 3"]
    )
    assert isinstance(cloud_potential.wrapped_coordinate_list_in_pixels, CoordinateList)
    assert isinstance(
        cloud_potential.wrapped_coordinate_list_in_pixels.get(), Float[Array, "_ 3"]
    )

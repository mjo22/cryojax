import jax.numpy as jnp
from jaxtyping import Array, Float

import cryojax.simulator as cs


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

    assert isinstance(
        fourier_potential.frequency_slice_in_pixels,
        Float[Array, "1 _ _ 3"],
    )
    assert isinstance(real_potential.coordinate_grid_in_pixels, Float[Array, "_ _ _ 3"])
    assert isinstance(cloud_potential.coordinate_list_in_pixels, Float[Array, "_ 3"])

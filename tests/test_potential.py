from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import cryojax.simulator as cs
from cryojax.coordinates import (
    AbstractCoordinates,
    CoordinateGrid,
    CoordinateList,
    FrequencySlice,
)
from cryojax.typing import PointCloudCoords3D, VolumeCoords, VolumeSliceCoords


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
        fourier_potential.wrapped_frequency_slice_in_pixels, FrequencySlice
    )
    assert isinstance(
        fourier_potential.wrapped_frequency_slice_in_pixels.get(), VolumeSliceCoords
    )
    assert isinstance(real_potential.wrapped_coordinate_grid_in_pixels, CoordinateGrid)
    assert isinstance(
        real_potential.wrapped_coordinate_grid_in_pixels.get(), VolumeCoords
    )
    assert isinstance(cloud_potential.wrapped_coordinate_list_in_pixels, CoordinateList)
    assert isinstance(
        cloud_potential.wrapped_coordinate_list_in_pixels.get(), PointCloudCoords3D
    )


def test_electron_potential_vmap(potential, integrator, config):
    filter_spec = jtu.tree_map(
        lambda x: not isinstance(x, AbstractCoordinates),
        potential,
        is_leaf=lambda x: isinstance(x, AbstractCoordinates),
    )
    # Add a leading dimension to scattering potential leaves
    potential = jtu.tree_map(
        lambda spec, x: jnp.expand_dims(x, axis=0) if spec else x,
        filter_spec,
        potential,
        is_leaf=lambda x: isinstance(x, AbstractCoordinates),
    )
    vmap, novmap = eqx.partition(potential, filter_spec)

    @partial(jax.vmap, in_axes=[0, None, None, None])
    def compute_image_stack(vmap, novmap, integrator, config):
        potential = eqx.combine(vmap, novmap)
        return integrator(potential, config)

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap, integrator, config)
    assert image_stack.shape[:1] == (1,)


def test_electron_potential_vmap_with_pipeline(potential, pose, integrator, config):
    pipeline = cs.ImagePipeline(config, cs.Specimen(potential, integrator, pose))

    def is_potential_leaves_without_coordinates(element):
        if isinstance(element, cs.AbstractScatteringPotential):
            return jtu.tree_map(
                lambda x: not isinstance(x, AbstractCoordinates),
                potential,
                is_leaf=lambda x: isinstance(x, AbstractCoordinates),
            )
        else:
            return False

    # Get filter spec for scattering potential
    filter_spec = jtu.tree_map(
        is_potential_leaves_without_coordinates,
        pipeline,
        is_leaf=lambda x: isinstance(x, cs.AbstractScatteringPotential),
    )
    # Add a leading dimension to scattering potential leaves
    pipeline = jtu.tree_map(
        lambda spec, x: jnp.expand_dims(x, axis=0) if spec else x,
        filter_spec,
        pipeline,
    )
    vmap, novmap = eqx.partition(pipeline, filter_spec)

    @partial(jax.vmap, in_axes=[0, None])
    def compute_image_stack(vmap, novmap):
        pipeline = eqx.combine(vmap, novmap)
        return pipeline.render()

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap)
    assert image_stack.shape[:1] == (1,)

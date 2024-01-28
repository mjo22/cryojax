import pytest

from functools import partial

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from cryojax.coordinates import (
    AbstractCoordinates,
    FrequencySlice,
    CoordinateGrid,
    CoordinateList,
    get_not_coordinate_filter_spec,
)
import cryojax.simulator as cs

from cryojax.typing import VolumeSliceCoords, VolumeCoords, CloudCoords3D


def test_voxel_electron_density_loaders():
    density_grid = jnp.zeros((10, 10, 10), dtype=float)
    voxel_size = 1.1
    fourier_density = cs.FourierVoxelGrid.from_density_grid(
        density_grid, voxel_size=voxel_size
    )
    real_density = cs.RealVoxelGrid.from_density_grid(
        density_grid, voxel_size=voxel_size
    )
    cloud_density = cs.VoxelCloud.from_density_grid(
        density_grid, voxel_size=voxel_size
    )
    for density in [real_density, fourier_density, cloud_density]:
        assert density.voxel_size == jnp.asarray(voxel_size)

    assert isinstance(fourier_density.frequency_slice, FrequencySlice)
    assert isinstance(fourier_density.frequency_slice.get(), VolumeSliceCoords)
    assert isinstance(real_density.coordinate_grid, CoordinateGrid)
    assert isinstance(real_density.coordinate_grid.get(), VolumeCoords)
    assert isinstance(cloud_density.coordinate_list, CoordinateList)
    assert isinstance(cloud_density.coordinate_list.get(), CloudCoords3D)


def test_electron_density_vmap(density, scattering, pose):
    filter_spec = get_not_coordinate_filter_spec(density)
    # Add a leading dimension to ElectronDensity leaves
    density = jtu.tree_map(
        lambda spec, x: jnp.expand_dims(x, axis=0) if spec else x,
        filter_spec,
        density,
        is_leaf=lambda x: isinstance(x, AbstractCoordinates),
    )
    vmap, novmap = eqx.partition(density, filter_spec)

    @partial(jax.vmap, in_axes=[0, None, None, None])
    def compute_image_stack(vmap, novmap, scattering, pose):
        density = eqx.combine(vmap, novmap)
        return scattering(cs.Specimen(density, pose))

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap, scattering, pose)
    assert image_stack.shape[:1] == (1,)


def test_electron_density_vmap_with_pipeline(density, pose, scattering):
    pipeline = cs.ImagePipeline(cs.Specimen(density, pose), scattering)
    # Get filter spec for ElectronDensity
    filter_spec = jtu.tree_map(
        cs.is_density_leaves_without_coordinates,
        pipeline,
        is_leaf=lambda x: isinstance(x, cs.AbstractElectronDensity),
    )
    # Add a leading dimension to ElectronDensity leaves
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

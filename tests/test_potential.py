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
)
import cryojax.simulator as cs

from cryojax.typing import VolumeSliceCoords, VolumeCoords, CloudCoords3D


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

    assert isinstance(fourier_potential.wrapped_frequency_slice, FrequencySlice)
    assert isinstance(
        fourier_potential.wrapped_frequency_slice.get(), VolumeSliceCoords
    )
    assert isinstance(real_potential.wrapped_coordinate_grid, CoordinateGrid)
    assert isinstance(real_potential.wrapped_coordinate_grid.get(), VolumeCoords)
    assert isinstance(cloud_potential.wrapped_coordinate_list, CoordinateList)
    assert isinstance(cloud_potential.wrapped_coordinate_list.get(), CloudCoords3D)


def test_electron_potential_vmap(potential, integrator, pose):
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
    def compute_image_stack(vmap, novmap, integrator, pose):
        potential = eqx.combine(vmap, novmap)
        return integrator(cs.Specimen(potential, pose).potential_in_lab_frame)

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap, integrator, pose)
    assert image_stack.shape[:1] == (1,)


def test_electron_potential_vmap_with_pipeline(potential, pose, integrator):
    pipeline = cs.ImagePipeline(cs.Specimen(potential, pose), integrator)

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

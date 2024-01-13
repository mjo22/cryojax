import pytest

from functools import partial

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cryojax.core as cc
import cryojax.simulator as cs

from cryojax.io import load_mrc
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
        assert density.n_stacked_dims == 0
        assert density.voxel_size == jnp.asarray(voxel_size)

    assert isinstance(fourier_density.frequency_slice, cc.FrequencySlice)
    assert isinstance(fourier_density.frequency_slice.get(), VolumeSliceCoords)
    assert isinstance(real_density.coordinate_grid, cc.CoordinateGrid)
    assert isinstance(real_density.coordinate_grid.get(), VolumeCoords)
    assert isinstance(cloud_density.coordinate_list, cc.CoordinateList)
    assert isinstance(cloud_density.coordinate_list.get(), CloudCoords3D)


@pytest.mark.parametrize(
    "stack_shape,voxel_size,n_stacked_dims",
    [((2,), 1.1, 1), ((2, 3), 1.1, 2), ((2, 2), jnp.full((2, 2), 1.1), 2)],
)
def test_voxel_electron_density_stacked_loader(
    stack_shape, voxel_size, n_stacked_dims
):
    density_grid = jnp.zeros((*stack_shape, 10, 10, 10), dtype=float)
    fourier_density = cs.FourierVoxelGrid.from_density_grid(
        density_grid,
        voxel_size=voxel_size,
        n_stacked_dims=n_stacked_dims,
    )
    real_density = cs.RealVoxelGrid.from_density_grid(
        density_grid,
        voxel_size=voxel_size,
        n_stacked_dims=n_stacked_dims,
    )
    for density in [real_density, fourier_density]:
        assert density.n_stacked_dims == n_stacked_dims
        assert density.weights.shape[: density.n_stacked_dims] == stack_shape
        assert density.stack_shape == stack_shape
        np.testing.assert_allclose(
            density.voxel_size, jnp.full(stack_shape, voxel_size)
        )


def test_electron_density_indexing(density):
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    assert stacked_density.n_stacked_dims == 1
    assert stacked_density[0].n_stacked_dims == 0
    assert stacked_density[:-1].n_stacked_dims == 1

    double_stacked_density = cls.from_list([stacked_density for _ in range(3)])
    assert double_stacked_density.n_stacked_dims == 2
    assert double_stacked_density[:, :].n_stacked_dims == 2
    assert double_stacked_density[:1].n_stacked_dims == 2
    assert double_stacked_density[0].n_stacked_dims == 1
    assert double_stacked_density[:, 0].n_stacked_dims == 1
    assert double_stacked_density[0, 0].n_stacked_dims == 0


def test_electron_density_shape(density):
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    double_stacked_density = cls.from_list([stacked_density for _ in range(2)])
    assert stacked_density.stack_shape == (3,)
    assert double_stacked_density.stack_shape == (2, 3)
    assert len(stacked_density) == 3
    assert len(double_stacked_density) == 6


def test_electron_density_vmap(density, scattering):
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    to_vmap = cc.get_not_coordinate_filter_spec(stacked_density)
    vmap, novmap = eqx.partition(stacked_density, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_image_stack(vmap, novmap, scattering):
        density = eqx.combine(vmap, novmap)
        return scattering(density)

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap, scattering)
    assert image_stack.shape[:1] == stacked_density.stack_shape
    assert image_stack.shape[0] == len(stacked_density)


def test_electron_density_vmap_with_pipeline(density, pose, scattering):
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    pipeline = cs.ImagePipeline(cs.Ensemble(stacked_density, pose), scattering)
    to_vmap = jtu.tree_map(
        cs.is_density_leaves,
        pipeline,
        is_leaf=lambda x: isinstance(x, cs.ElectronDensity),
    )
    vmap, novmap = eqx.partition(pipeline, to_vmap)

    @partial(jax.vmap, in_axes=[0, None])
    def compute_image_stack(vmap, novmap):
        pipeline = eqx.combine(vmap, novmap)
        return pipeline.render()

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap)
    assert image_stack.shape[:1] == stacked_density.stack_shape
    assert image_stack.shape[0] == len(stacked_density)

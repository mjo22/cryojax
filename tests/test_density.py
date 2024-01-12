import pytest

from functools import partial

import jax
import jax.tree_util as jtu
import equinox as eqx
import cryojax.core as cc


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
    to_vmap = jtu.tree_map(
        cc.is_not_coordinate_array, stacked_density, is_leaf=cc.is_coordinates
    )
    vmap, novmap = eqx.partition(stacked_density, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_image_stack(vmap, novmap, scattering):
        density = eqx.combine(vmap, novmap)
        return scattering(density)

    # vmap over first axis
    image_stack = compute_image_stack(vmap, novmap, scattering)
    assert image_stack.shape[:1] == stacked_density.stack_shape
    assert image_stack.shape[0] == len(stacked_density)

import pytest

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from cryojax.simulator import Ensemble, Conformation


def test_conformation(density, pose, scattering):
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    ensemble = Ensemble(stacked_density, pose, conformation=Conformation(0))
    _ = scattering(ensemble.density_at_conformation_and_pose, pose)


def test_conformation_vmap(density, pose, scattering):
    # Build Ensemble
    cls = type(density)
    stacked_density = cls.from_list([density for _ in range(3)])
    ensemble = Ensemble(
        stacked_density,
        pose,
        conformation=Conformation(jnp.asarray((0, 1, 2, 1, 0))),
    )
    # Setup vmap
    is_vmap = lambda x: isinstance(x, Conformation)
    to_vmap = jtu.tree_map(is_vmap, ensemble, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(ensemble, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None, None])
    def compute_conformation_stack(vmap, novmap, scattering, pose):
        ensemble = eqx.combine(vmap, novmap)
        return scattering(ensemble.get_density(), pose)

    # Vmap over conformations
    image_stack = compute_conformation_stack(vmap, novmap, scattering, pose)
    assert image_stack.shape[0] == ensemble.conformation.get().shape[0]

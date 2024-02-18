import pytest

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from cryojax.simulator import DiscreteEnsemble, DiscreteConformation


def test_conformation(potential, pose, integrator):
    potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(potential, pose, conformation=DiscreteConformation(0))
    _ = integrator.scatter_to_exit_plane(ensemble)


def test_conformation_vmap(potential, pose, integrator):
    # Build Ensemble
    cls = type(potential)
    stacked_potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(
        stacked_potential,
        pose,
        conformation=DiscreteConformation(jnp.asarray((0, 1, 2, 1, 0))),
    )
    # Setup vmap
    is_vmap = lambda x: isinstance(x, DiscreteConformation)
    to_vmap = jtu.tree_map(is_vmap, ensemble, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(ensemble, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_conformation_stack(vmap, novmap, integrator):
        ensemble = eqx.combine(vmap, novmap)
        return integrator.scatter_to_exit_plane(ensemble)

    # Vmap over conformations
    image_stack = compute_conformation_stack(vmap, novmap, integrator)
    assert image_stack.shape[0] == ensemble.conformation.get().shape[0]

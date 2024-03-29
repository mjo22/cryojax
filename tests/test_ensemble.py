from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from cryojax.simulator import DiscreteConformation, DiscreteEnsemble, Instrument


def test_conformation(potential, pose, integrator, config):
    potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(
        potential, integrator, pose, conformation=DiscreteConformation(0)
    )
    _ = ensemble.scatter_to_exit_plane(config)


def test_conformation_vmap(potential, pose, integrator, config):
    # Build Ensemble
    stacked_potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(
        stacked_potential,
        integrator,
        pose,
        conformation=DiscreteConformation(jnp.asarray((0, 1, 2, 1, 0))),
    )
    # Setup vmap
    is_vmap = lambda x: isinstance(x, DiscreteConformation)
    to_vmap = jtu.tree_map(is_vmap, ensemble, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(ensemble, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_conformation_stack(vmap, novmap, config):
        ensemble = eqx.combine(vmap, novmap)
        instrument = Instrument(300.0)
        return ensemble.scatter_to_exit_plane(instrument, config)

    # Vmap over conformations
    image_stack = compute_conformation_stack(vmap, novmap, config)
    assert image_stack.shape[0] == ensemble.conformation.value.shape[0]

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import cryojax.simulator as cxs
from cryojax.simulator import DiscreteConformation, DiscreteEnsemble, Instrument


def test_conformation(potential, pose, projection_method, transfer_theory, config):
    potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(potential, pose, conformation=DiscreteConformation(0))
    instrument = Instrument(300.0)
    theory = cxs.LinearScatteringTheory(ensemble, projection_method, transfer_theory)
    _ = theory.compute_fourier_phase_shifts_at_exit_plane(config, instrument)


def test_conformation_vmap(potential, pose, projection_method, transfer_theory, config):
    # Build Ensemble
    state_space = tuple([potential for _ in range(3)])
    ensemble = DiscreteEnsemble(
        state_space,
        pose,
        conformation=jax.vmap(lambda value: DiscreteConformation(value))(
            jnp.asarray((0, 1, 2, 1, 0))
        ),
    )
    theory = cxs.LinearScatteringTheory(ensemble, projection_method, transfer_theory)
    # Setup vmap
    is_vmap = lambda x: isinstance(x, DiscreteConformation)
    to_vmap = jtu.tree_map(is_vmap, theory, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(theory, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_conformation_stack(vmap, novmap, config):
        theory = eqx.combine(vmap, novmap)
        instrument = Instrument(300.0)
        return theory.compute_fourier_phase_shifts_at_exit_plane(config, instrument)

    # Vmap over conformations
    image_stack = compute_conformation_stack(vmap, novmap, config)
    assert image_stack.shape[0] == ensemble.conformation.value.shape[0]

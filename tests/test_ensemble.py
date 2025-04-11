from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import cryojax.simulator as cxs
from cryojax.simulator import DiscreteConformationalVariable, DiscreteStructuralEnsemble


def test_conformation(potential, pose, projection_method, transfer_theory, config):
    potential = tuple([potential for _ in range(3)])
    ensemble = DiscreteStructuralEnsemble(
        potential, pose, conformation=DiscreteConformationalVariable(0)
    )
    theory = cxs.WeakPhaseScatteringTheory(ensemble, projection_method, transfer_theory)
    _ = theory.compute_object_spectrum_at_exit_plane(config)


def test_conformation_vmap(potential, pose, projection_method, transfer_theory, config):
    # Build Ensemble
    state_space = tuple([potential for _ in range(3)])
    ensemble = DiscreteStructuralEnsemble(
        state_space,
        pose,
        conformation=jax.vmap(lambda value: DiscreteConformationalVariable(value))(
            jnp.asarray((0, 1, 2, 1, 0))
        ),
    )
    theory = cxs.WeakPhaseScatteringTheory(ensemble, projection_method, transfer_theory)
    # Setup vmap
    is_vmap = lambda x: isinstance(x, DiscreteConformationalVariable)
    to_vmap = jtu.tree_map(is_vmap, theory, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(theory, to_vmap)

    @partial(jax.vmap, in_axes=[0, None, None])
    def compute_conformation_stack(vmap, novmap, config):
        theory = eqx.combine(vmap, novmap)
        return theory.compute_object_spectrum_at_exit_plane(config)

    # Vmap over conformations
    image_stack = compute_conformation_stack(vmap, novmap, config)
    assert image_stack.shape[0] == ensemble.conformation.value.shape[0]

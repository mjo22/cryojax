import pytest

import cryojax.simulator as cs
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import jax.random as jr


def test_custom_variance(
    likelihood_model,
    likelihood_model_with_custom_variance,
    test_image,
):
    likelihood_model = eqx.tree_at(
        lambda m: m.pipeline.solvent, likelihood_model, cs.NullIce()
    )
    np.testing.assert_allclose(
        likelihood_model.variance,
        likelihood_model_with_custom_variance.variance,
    )
    np.testing.assert_allclose(
        likelihood_model.log_probability(test_image),
        likelihood_model_with_custom_variance.log_probability(test_image),
    )
    np.testing.assert_allclose(
        likelihood_model.sample(jr.PRNGKey(seed=0)),
        likelihood_model_with_custom_variance.sample(jr.PRNGKey(seed=0)),
    )

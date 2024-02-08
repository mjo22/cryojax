import pytest

from cryojax.inference import distributions as dist
from cryojax.image import operators as op
import cryojax.simulator as cs
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr


def test_custom_variance(noisy_model, config, test_image):
    noisy_model = eqx.tree_at(
        lambda m: (m.solvent, m.instrument.detector),
        noisy_model,
        (cs.NullIce(), cs.GaussianDetector(op.Constant(1.0))),
    )
    likelihood_model = dist.IndependentFourierGaussian(noisy_model)
    likelihood_model_with_custom_variance = dist.IndependentFourierGaussian(
        noisy_model, variance=op.Constant(1.0)
    )
    freqs = config.frequency_grid_in_angstroms.get()
    assert eqx.tree_equal(
        likelihood_model.variance,
        likelihood_model_with_custom_variance.variance,
    )
    np.testing.assert_allclose(
        likelihood_model.variance(freqs),
        likelihood_model_with_custom_variance.variance(freqs),
    )
    np.testing.assert_allclose(
        likelihood_model.log_probability(test_image),
        likelihood_model_with_custom_variance.log_probability(test_image),
    )
    np.testing.assert_allclose(
        likelihood_model.sample(jr.PRNGKey(seed=0)),
        likelihood_model_with_custom_variance.sample(jr.PRNGKey(seed=0)),
    )

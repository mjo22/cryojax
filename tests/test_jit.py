import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import cryojax.simulator as cs
from jax import config

from functools import partial

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["likelihood_model"])
def test_jit(model, test_image, request):
    likelihood_model = request.getfixturevalue(model)
    key = jr.PRNGKey(0)

    @jax.jit
    def compute_image(model, key):
        return model.sample(key)

    @jax.jit
    def compute_loss(model, test_image):
        return model.log_probability(test_image)

    np.testing.assert_allclose(
        compute_image(likelihood_model, key), likelihood_model.sample(key)
    )
    np.testing.assert_allclose(
        likelihood_model.log_probability(test_image),
        compute_loss(likelihood_model, test_image),
    )


def test_value_and_grad(likelihood_model, test_image):
    def build_model(model, params):
        where = lambda m: m.pipeline.specimen.pose.offset_z
        return eqx.tree_at(where, model, params["offset_z"])

    @jax.jit
    @partial(jax.value_and_grad, argnums=1)
    def compute_loss(model, params, test_image):
        model = build_model(model, params)
        return model.log_probability(test_image)

    value, grad = compute_loss(
        likelihood_model, dict(offset_z=jnp.asarray(100.0)), test_image
    )

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import cryojax.simulator as cs

from functools import partial

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model"])
def test_jit(model, test_image, request):
    model = request.getfixturevalue(model)
    key = jr.PRNGKey(0)

    @jax.jit
    def compute_image(model, key):
        return model.sample(key)

    np.testing.assert_allclose(compute_image(model, key), model.sample(key))

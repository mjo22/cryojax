import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import cryojax.simulator as cs


@pytest.mark.parametrize("model", ["noisy_model"])
def test_filters_and_masks(
    model, filtered_and_masked_model, request, filters, masks
):
    """Make sure that adding null filters and masks does not change output"""
    model = request.getfixturevalue(model)
    # Add null filters and masks
    null_mask = eqx.tree_at(lambda m: m.mask, masks, jnp.asarray(1.0))
    null_filter = eqx.tree_at(lambda f: f.filter, filters, jnp.asarray(1.0))
    where = lambda m: (m.filter, m.mask)
    model_with_null_mask = eqx.tree_at(
        where, filtered_and_masked_model, (None, null_mask)
    )
    model_with_null_filter = eqx.tree_at(
        where, filtered_and_masked_model, (null_filter, None)
    )
    model_with_null_filter_and_mask = eqx.tree_at(
        where,
        filtered_and_masked_model,
        (null_filter, null_mask),
    )
    # Compute images
    key = jax.random.PRNGKey(0)
    image = model.render()
    noisy_image = model.sample(key)
    # Check render
    np.testing.assert_allclose(model_with_null_mask.render(), image)
    np.testing.assert_allclose(model_with_null_filter.render(), image)
    np.testing.assert_allclose(model_with_null_filter_and_mask.render(), image)
    # Check sample
    np.testing.assert_allclose(model_with_null_mask.sample(key), noisy_image)
    np.testing.assert_allclose(model_with_null_filter.sample(key), noisy_image)
    np.testing.assert_allclose(
        model_with_null_filter_and_mask.sample(key), noisy_image
    )

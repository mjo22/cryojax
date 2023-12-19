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


def test_jit(
    instrument, scattering, weights_and_coordinates, resolution, test_image
):
    """
    Test the jit pipeline, without equinox. This requires building
    the ElectronDensity with the voxel density as input. Otherwise,
    jit compilation time will be very long.

    To avoid this awkward step, use equinox.
    """

    key = jr.split(jr.PRNGKey(0))

    def build_specimen(voxels):
        density = cs.VoxelGrid(
            weights=voxels["weights"], coordinates=voxels["coordinates"]
        )
        return cs.Specimen(density=density, resolution=resolution)

    def build_model(voxels):
        specimen = build_specimen(voxels)
        return cs.GaussianImage(
            specimen=specimen,
            scattering=scattering,
            instrument=instrument,
        )

    @jax.jit
    def compute_image(voxels):
        model = build_model(voxels)
        return model.sample(key)

    @jax.jit
    def compute_loss(voxels, test_image):
        model = build_model(voxels)
        return model.log_probability(test_image)

    unjitted_model = build_model(weights_and_coordinates)
    np.testing.assert_allclose(
        compute_image(weights_and_coordinates), unjitted_model.sample(key)
    )
    np.testing.assert_allclose(
        compute_loss(weights_and_coordinates, test_image),
        unjitted_model.log_probability(test_image),
    )


def test_equinox_jit(likelihood_model, test_image):
    key = jr.split(jr.PRNGKey(0))

    @eqx.filter_jit
    def compute_image(model):
        return model.sample(key)

    @eqx.filter_jit
    def compute_loss(model, test_image):
        return model.log_probability(test_image)

    np.testing.assert_allclose(
        compute_image(likelihood_model), likelihood_model.sample(key)
    )
    np.testing.assert_allclose(
        compute_loss(likelihood_model, test_image),
        likelihood_model.log_probability(test_image),
    )


def test_equinox_value_and_grad(likelihood_model, test_image):
    def build_model(model, params):
        where = lambda m: m.specimen.pose.offset_z
        return eqx.tree_at(where, model, params["offset_z"])

    @jax.jit
    @partial(jax.value_and_grad, argnums=1)
    def compute_loss(model, params, test_image):
        model = build_model(model, params)
        return model.log_probability(test_image)

    value, grad = compute_loss(
        likelihood_model, dict(offset_z=jnp.asarray(100.0)), test_image
    )

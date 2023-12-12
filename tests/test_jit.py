import pytest

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cryojax.simulator as cs

from functools import partial


def test_jit(
    state, scattering, weights_and_coordinates, resolution, test_image
):
    """
    Test the jit pipeline, without equinox. This requires building
    the ElectronDensity with the voxel density as input. Otherwise,
    jit compilation time will be very long.

    To avoid this awkward step, use equinox.
    """

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
            state=state,
            observed=test_image,
        )

    @jax.jit
    def compute_image(voxels):
        model = build_model(voxels)
        return model.sample()

    @jax.jit
    def compute_loss(voxels):
        model = build_model(voxels)
        return model.log_probability()

    unjitted_model = build_model(weights_and_coordinates)
    np.testing.assert_allclose(
        compute_image(weights_and_coordinates), unjitted_model.sample()
    )
    np.testing.assert_allclose(
        compute_loss(weights_and_coordinates), unjitted_model.log_probability()
    )


def test_equinox_jit(likelihood_model):
    @eqx.filter_jit
    def compute_image(model):
        return model.sample()

    @eqx.filter_jit
    def compute_loss(model):
        return model()

    np.testing.assert_allclose(
        compute_image(likelihood_model), likelihood_model.sample()
    )
    np.testing.assert_allclose(
        compute_loss(likelihood_model), likelihood_model()
    )


def test_equinox_value_and_grad(likelihood_model):
    def build_model(model, params):
        where = lambda m: m.state.pose.offset_z
        return eqx.tree_at(where, model, params["offset_z"])

    @jax.jit
    @partial(jax.value_and_grad, argnums=1)
    def compute_loss(model, params):
        model = build_model(model, params)
        return model()

    value, grad = compute_loss(
        likelihood_model, dict(offset_z=jnp.asarray(100.0))
    )

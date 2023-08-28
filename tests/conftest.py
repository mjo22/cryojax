import os
import pytest

from jax import random
from jax import config

import cryojax.simulator as cs

config.update("jax_enable_x64", False)


@pytest.fixture
def setup():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_13pf_bfm1_ps5_28.mrc"
    )
    scattering = cs.FourierSliceScattering(shape=(81, 81))
    specimen = cs.ElectronGrid.from_file(filename, resolution=5.32)
    # scattering = cs.NufftScattering(shape=(81, 81), eps=1e-4)
    # specimen = cs.ElectronCloud.from_file(filename, resolution=5.32)
    return scattering, specimen


@pytest.fixture
def filters(setup):
    scattering, _ = setup
    return [cs.LowpassFilter(scattering.padded_shape)]
    # return []


@pytest.fixture
def masks(setup):
    scattering, _ = setup
    return [cs.CircularMask(scattering.shape)]


@pytest.fixture
def state():
    return cs.PipelineState(
        pose=cs.EulerPose(degrees=False),
        ice=cs.ExponentialNoiseIce(key=random.PRNGKey(seed=1)),
        optics=cs.CTFOptics(),
        exposure=cs.UniformExposure(),
        detector=cs.WhiteNoiseDetector(
            pixel_size=5.32, key=random.PRNGKey(seed=0)
        ),
    )


@pytest.fixture
def scattering_model(setup, state, filters, masks):
    scattering, specimen = setup
    return cs.ScatteringImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        masks=masks,
        filters=filters,
    )


@pytest.fixture
def optics_model(setup, state, filters, masks):
    scattering, specimen = setup
    return cs.OpticsImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
        masks=masks,
    )


@pytest.fixture
def noisy_model(setup, state, filters, masks):
    scattering, specimen = setup
    return cs.GaussianImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
        masks=masks,
    )


@pytest.fixture
def maskless_model(setup, state, filters):
    scattering, specimen = setup
    return cs.OpticsImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
    )

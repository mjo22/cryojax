import os
import pytest

from jax import random
from jax import config

import cryojax.simulator as cs

config.update("jax_enable_x64", True)


@pytest.fixture
def scattering():
    # return cs.NufftScattering(shape=(81, 81), eps=1e-4)
    return cs.FourierSliceScattering(shape=(81, 81))


@pytest.fixture
def density():
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_monomer_bfm1_ps5_28.mrc"
    )
    # return cs.ElectronCloud.from_file(filename, resolution=5.32)
    return cs.ElectronGrid.from_file(filename)


@pytest.fixture
def filters(scattering):
    return [cs.LowpassFilter(scattering.padded_shape)]
    # return []


@pytest.fixture
def masks(scattering):
    return [cs.CircularMask(scattering.shape)]


@pytest.fixture
def state():
    return cs.PipelineState(
        pose=cs.EulerPose(degrees=False),
        ice=cs.GaussianIce(key=random.PRNGKey(seed=1)),
        optics=cs.CTFOptics(),
        exposure=cs.UniformExposure(N=1e5, mu=1.0),
        detector=cs.GaussianDetector(
            pixel_size=5.32, key=random.PRNGKey(seed=0)
        ),
    )


@pytest.fixture
def specimen(density):
    return cs.Specimen(density=density, resolution=5.32)


@pytest.fixture
def scattering_model(scattering, specimen, state, filters, masks):
    return cs.ScatteringImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        masks=masks,
        filters=filters,
    )


@pytest.fixture
def optics_model(scattering, specimen, state, filters, masks):
    return cs.OpticsImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
        masks=masks,
    )


@pytest.fixture
def noisy_model(scattering, specimen, state, filters, masks):
    return cs.GaussianImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
        masks=masks,
    )


@pytest.fixture
def maskless_model(scattering, specimen, state, filters):
    return cs.OpticsImage(
        scattering=scattering,
        specimen=specimen,
        state=state,
        filters=filters,
    )

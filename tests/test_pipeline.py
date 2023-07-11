import os
import pytest

from jax_2dtm.io import load_grid_as_cloud

from jax_2dtm.simulator import ScatteringConfig
from jax_2dtm.simulator import EulerPose, ParameterState
from jax_2dtm.simulator import ScatteringImage


@pytest.fixture
def model():
    # Read in cloud and set config
    filename = os.path.join(
        os.path.dirname(__file__), "data", "3jar_13pf_bfm1_ps5_28.mrc"
    )
    config = ScatteringConfig((80, 80), 5.32, eps=1e-4)
    cloud = load_grid_as_cloud(filename, config)
    # Setup and compute model
    state = ParameterState(pose=EulerPose())
    return ScatteringImage(
        config=config, cloud=cloud, state=state
    )  # , filters=[])


def test_shape(model):
    image = model()
    assert image.shape == model.config.shape

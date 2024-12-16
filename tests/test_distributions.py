import numpy as np
import pytest

import cryojax.simulator as cxs
from cryojax.inference import distributions as dist


@pytest.mark.parametrize(
    "cls, scattering_theory, instrument_config",
    [
        (dist.IndependentGaussianPixels, "theory", "config"),
        (dist.IndependentGaussianFourierModes, "theory", "config"),
    ],
)
def test_simulate_signal_from_gaussian_distributions(
    cls, scattering_theory, instrument_config, request
):
    scattering_theory = request.getfixturevalue(scattering_theory)
    instrument_config = request.getfixturevalue(instrument_config)
    imaging_pipeline = cxs.ContrastImagingPipeline(instrument_config, scattering_theory)
    distribution = cls(imaging_pipeline)
    np.testing.assert_allclose(imaging_pipeline.render(), distribution.compute_signal())

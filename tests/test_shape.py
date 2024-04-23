import jax
import numpy as np
import pytest

import cryojax.simulator as cs
from cryojax.data import read_array_with_spacing_from_mrc
from cryojax.image import crop_to_shape


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["noisy_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model.render()
    padded_image = model.render(postprocess=False)
    assert image.shape == model.instrument_config.shape
    assert padded_image.shape == model.instrument_config.padded_shape


@pytest.mark.parametrize("model", ["noisy_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model.render(get_real=False)
    padded_image = model.render(postprocess=False, get_real=False)
    assert (
        image.shape
        == model.instrument_config.wrapped_frequency_grid_in_pixels.get().shape[0:2]
    )
    assert (
        padded_image.shape
        == model.instrument_config.wrapped_padded_frequency_grid_in_pixels.get().shape[
            0:2
        ]
    )


@pytest.mark.parametrize("shape", [(65, 65), (65, 64), (64, 65)])
def test_even_vs_odd_image_shape(shape, sample_mrc_path, pixel_size):
    control_shape = (64, 64)
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    potential = cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size
    )
    assert control_shape == potential.fourier_voxel_grid.shape[0:2]
    pose = cs.EulerAnglePose()
    method = cs.FourierSliceExtraction()
    specimen = cs.SingleStructureEnsemble(potential, pose)
    transfer_theory = cs.ContrastTransferTheory(cs.ContrastTransferFunction())
    theory = cs.LinearScatteringTheory(specimen, method, transfer_theory)
    config_control = cs.InstrumentConfig(
        control_shape, pixel_size, voltage_in_kilovolts=300.0
    )
    config_test = cs.InstrumentConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    pipeline_control = cs.ContrastImagingPipeline(config_control, theory)
    pipeline_test = cs.ContrastImagingPipeline(config_test, theory)

    np.testing.assert_allclose(
        crop_to_shape(pipeline_test.render(), control_shape),
        pipeline_control.render(),
        atol=1e-4,
    )

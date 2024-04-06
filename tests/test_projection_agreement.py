import jax
import numpy as np
import pytest

import cryojax.simulator as cs
from cryojax.data import read_array_with_spacing_from_mrc
from cryojax.image import crop_to_shape


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("shape", [(65, 65), (65, 64), (64, 65)])
def test_even_vs_odd_image_shape(shape, sample_mrc_path, pixel_size):
    control_shape = (64, 64)
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    potential = cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size
    )
    assert control_shape == potential.fourier_voxel_grid.shape[0:2]
    pose = cs.EulerAnglePose()
    integrator = cs.FourierSliceExtract()
    specimen = cs.Specimen(potential, integrator, pose)
    config_control = cs.ImageConfig(control_shape, pixel_size)
    config_test = cs.ImageConfig(shape, pixel_size)
    instrument = cs.Instrument(voltage_in_kilovolts=300.0)
    pipeline_control = cs.ImagePipeline(config_control, specimen, instrument)
    pipeline_test = cs.ImagePipeline(config_test, specimen, instrument)

    np.testing.assert_allclose(
        crop_to_shape(pipeline_test.render(), control_shape),
        pipeline_control.render(),
    )

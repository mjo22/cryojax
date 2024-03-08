import pytest

import numpy as np
import cryojax.simulator as cs
from cryojax.image import crop_to_shape
from cryojax.io import read_volume_with_voxel_size_from_mrc


@pytest.mark.parametrize("shape", [(65, 65), (65, 64), (64, 65)])
def test_even_vs_odd_image_shape(shape, sample_mrc_path, pixel_size):
    control_shape = (64, 64)
    real_voxel_grid, voxel_size = read_volume_with_voxel_size_from_mrc(sample_mrc_path)
    potential = cs.FourierVoxelGrid.from_real_voxel_grid(real_voxel_grid, voxel_size)
    assert control_shape == potential.fourier_voxel_grid.shape[0:2]
    pose = cs.EulerAnglePose()
    specimen = cs.Specimen(potential, pose)
    scattering_control = cs.FourierSliceExtract(
        cs.ImageConfig(control_shape, pixel_size)
    )
    scattering_test = cs.FourierSliceExtract(cs.ImageConfig(shape, pixel_size))
    pipeline_control = cs.ImagePipeline(specimen, scattering_control)
    pipeline_test = cs.ImagePipeline(specimen, scattering_test)

    np.testing.assert_allclose(
        pipeline_control.render(),
        crop_to_shape(pipeline_test.render(), control_shape),
    )

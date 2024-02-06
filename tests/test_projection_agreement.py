import pytest

import numpy as np
import cryojax.simulator as cs
from cryojax.image import crop_to_shape
from cryojax.io import read_image_or_volume_with_spacing_from_mrc


@pytest.mark.parametrize("shape", [(65, 65), (65, 64), (64, 65)])
def test_even_vs_odd_image_shape(shape, sample_mrc_path, pixel_size):
    control_shape = (64, 64)
    density_grid, voxel_size = read_image_or_volume_with_spacing_from_mrc(
        sample_mrc_path
    )
    density = cs.FourierVoxelGrid.from_density_grid(density_grid, voxel_size)
    assert control_shape == density.fourier_density_grid.shape[0:2]
    pose = cs.EulerPose()
    specimen = cs.Specimen(density, pose)
    scattering_control = cs.FourierSliceExtract(
        cs.ImageManager(control_shape, pixel_size)
    )
    scattering_test = cs.FourierSliceExtract(cs.ImageManager(shape, pixel_size))
    pipeline_control = cs.ImagePipeline(specimen, scattering_control)
    pipeline_test = cs.ImagePipeline(specimen, scattering_test)

    np.testing.assert_allclose(
        pipeline_control.render(),
        crop_to_shape(pipeline_test.render(), control_shape),
    )

import jax.numpy as jnp
import numpy as np
import pytest

## plotting settings
import cryojax.simulator as cxs
from cryojax.constants import convert_keV_to_angstroms
from cryojax.coordinates import make_coordinate_grid
from cryojax.image import irfftn


@pytest.mark.parametrize(
    "phi,theta,psi",
    [
        (0.0, 0.0, 0.0),
        # (30.0, 10.0, -10.0),
    ],
)
def test_fourier_slice_interpolation(phi, theta, psi):
    ## Create a single Gaussian centered in the middle
    sigma = 10.0
    npts = 101
    voxel_size = 1.0
    xyz = make_coordinate_grid((npts, npts, npts), voxel_size)
    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    data = np.exp(-(r**2) / (2 * sigma**2))
    print(data.shape)

    ## Plot the center slice
    # plt.imshow(data[:, :, 50])
    # plt.savefig("./plots/test_map_coordinates_0.png")

    # x_slice = np.linspace(-1, 1, 100)
    # y_slice = np.linspace(-1, 1, 100)

    # Instantiate API
    # pose = cxs.EulerAnglePose(view_phi=phi, view_theta=theta, view_psi=psi)
    voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(data, voxel_size)
    # voxel_potential_in_lab_frame = voxel_potential.rotate_to_pose(pose)
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    fourier_slice_with_zero_in_corner = (
        potential_integrator.extract_voxels_from_grid_points(
            voxel_potential.fourier_voxel_grid,
            voxel_potential.frequency_slice_in_pixels,
            voxel_potential.voxel_size,
            wavelength_in_angstroms=convert_keV_to_angstroms(300.0),
        )
    )
    _ = jnp.fft.fftshift(fourier_slice_with_zero_in_corner, axes=(0,))
    _ = irfftn(fourier_slice_with_zero_in_corner, s=(npts, npts))


if __name__ == "__main__":
    test_fourier_slice_interpolation(0.0, 0.0, 0.0)

import jax.numpy as jnp
import numpy as np
# import pytest

## plotting settings
import cryojax.simulator as cxs
from cryojax.constants import convert_keV_to_angstroms
from cryojax.coordinates import make_coordinate_grid, make_frequency_slice
from cryojax.image import irfftn

from matplotlib import pyplot as plt


# @pytest.mark.parametrize(
#     "phi,theta,psi",
#     [
#         (0.0, 0.0, 0.0),
#         # (30.0, 10.0, -10.0),
#     ],
# )
def test_fourier_slice_interpolation(phi, theta, psi):
    
    ## Create a single Gaussian centered in the middle for real space
    sigma = 10.0
    npts = 101
    voxel_size = 1.0
    xyz = make_coordinate_grid((npts, npts, npts), voxel_size)
    r = np.sqrt(xyz[:,:,:,0] ** 2 + xyz[:,:,:,1] ** 2 + xyz[:,:,:,2] ** 2)
    density_3d_true = np.exp(-(r**2) / (2 * sigma**2))

    # Instantiate API
    # pose = cxs.EulerAnglePose(view_phi=phi, view_theta=theta, view_psi=psi)
    frequency_slice = make_frequency_slice((npts, npts), voxel_size, half_space = False)
    print(frequency_slice.shape)
    voxel_potential = cxs.FourierVoxelGridPotential(density_3d_true, frequency_slice, voxel_size)
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
    image_fourier = jnp.fft.fftshift(fourier_slice_with_zero_in_corner, axes=(0,))
    # image_real = irfftn(fourier_slice_with_zero_in_corner, s=(npts, npts))
    # print(image_fourier.shape, image_fourier.dtype)
    
    # plt.imshow(image_fourier.real)
    # plt.savefig("./plots/test_map_coordinates_1.png")
    
    xyz_slice = voxel_potential.frequency_slice_in_pixels[0]
    # print(xyz_slice)
    # xyz_slice = xyz_slice.reshape(-1, 3)
    
    r_slice = np.sqrt(xyz_slice[:,:,0] ** 2 + xyz_slice[:,:,1] ** 2 + xyz_slice[:,:,2] ** 2)
    slice_density_true = np.exp(-(r_slice ** 2) / (2 * sigma**2))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.imshow(density_3d_true[:, :, 50])
    ax2 = fig.add_subplot(132)
    ax2.imshow(slice_density_true)
    ax3 = fig.add_subplot(133)
    ax3.imshow(image_fourier.real)
    plt.savefig("./plots/test_map_coordinates_2.png")
    
    


if __name__ == "__main__":
    test_fourier_slice_interpolation(0.0, 0.0, 0.0)

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

# import pytest
## plotting settings
import cryojax.simulator as cxs
from cryojax.constants import convert_keV_to_angstroms
from cryojax.coordinates import (
    make_coordinate_grid,
    make_frequency_grid,
    make_frequency_slice,
)
from cryojax.image import irfftn


# @pytest.mark.parametrize(
#     "phi,theta,psi",
#     [
#         (0.0, 0.0, 0.0),
#         # (30.0, 10.0, -10.0),
#     ],
# )
def test_fourier_slice_interpolation(phi, theta, psi):
    
    ###
    ### Currently, it looks more like test FFT then test interpolation...
    ###
    
    ##
    ## Create a single Gaussian centered in the middle for real space
    ##
    ## TODO: perturbed the gaussian with planewaves (i.e. real space translation)
    ## e.g. planewave_translation = jnp.exp(1j * (xyz @ k) * (- 2.0 * np.pi))
    ## or just with the planewave as density, because we want to test up to high frequencies
    ## so the exponential function is not covering up bugs at high frequencies
    ##
    sigma = 10.0 ## in Angstrom
    npts = 100  ## number of points in each dimension
    voxel_size = 1.0 ## in Angstrom
    xyz = make_coordinate_grid((npts, npts, npts), voxel_size)
    xyz_fourier = jnp.fft.fftshift(
        make_frequency_grid((npts, npts, npts), voxel_size, half_space=False)
    )
    r = jnp.linalg.norm(xyz, axis=-1)
    r_fourier = jnp.linalg.norm(xyz_fourier, axis=-1)
    physical_density_3d_true = jnp.exp(- (r ** 2) / (2 * sigma ** 2))
    physical_image_projected_true = jnp.sum(physical_density_3d_true, axis=-1)
    fourier_density_3d_true = jnp.exp(- (r_fourier ** 2) * (2 * np.pi ** 2 * sigma ** 2))
    
    ## setup the euler angles 
    
    # phi, theta, psi = 0.0, 0.0, 0.0
    # phi, theta, psi = 30.0, 10.0, -10.0
    # pose = cxs.EulerAnglePose(view_phi=phi, view_theta=theta, view_psi=psi)
    
    ## TODO: rotate the fourier slice by the euler angles
    ## TODO: analytical fourier slice density determined by the euler angles
    
    ## setup the fourier slices
    frequency_slice_in_pixels = make_frequency_slice((npts, npts), half_space=False)
    print(frequency_slice_in_pixels.shape)
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order = 1)
    fourier_slice_with_zero_in_corner = (
        potential_integrator.extract_voxels_from_grid_points(
            fourier_density_3d_true.astype(complex),
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms=convert_keV_to_angstroms(300.0),
        )
    )
    image_fourier = jnp.fft.fftshift(fourier_slice_with_zero_in_corner, axes=(0,))
    image_real = irfftn(fourier_slice_with_zero_in_corner, s=(npts, npts))
    # print(image_fourier.shape, image_fourier.dtype)

    xyz_slice = frequency_slice_in_pixels[0]
    # print(xyz_slice)
    # xyz_slice = xyz_slice.reshape(-1, 3)
    r_slice = np.sqrt(
        xyz_slice[:, :, 0] ** 2 + xyz_slice[:, :, 1] ** 2 + xyz_slice[:, :, 2] ** 2
    )
    slice_density_true = np.exp(- (r_slice ** 2) * (2 * np.pi ** 2 * sigma ** 2))

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(141)
    ax1.imshow(physical_image_projected_true)
    ax2 = fig.add_subplot(142)
    ax2.imshow(image_fourier.real)
    ax3 = fig.add_subplot(143)
    ax3.imshow(slice_density_true)
    ax4 = fig.add_subplot(144)
    ax4.imshow(image_real.real)
    plt.savefig("./plots/test_map_coordinates_2.png")
    
    normalized_physical_image_projected_true = physical_image_projected_true - np.mean(physical_image_projected_true)
    normalized_physical_image_projected_true = normalized_physical_image_projected_true / np.linalg.norm(normalized_physical_image_projected_true)
    normalized_image_real = image_real.real - np.mean(image_real.real)
    normalized_image_real = normalized_image_real / np.linalg.norm(normalized_image_real)
    error = np.linalg.norm(normalized_physical_image_projected_true - normalized_image_real)
    print("Error: ", error)


if __name__ == "__main__":
    test_fourier_slice_interpolation(0.0, 0.0, 0.0)

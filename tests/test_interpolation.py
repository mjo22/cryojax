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
    sigma_fourier = 1 / sigma
    npts = 101
    voxel_size = 1.0
    xyz = make_coordinate_grid((npts, npts, npts), voxel_size)
    xyz_fourier = jnp.fft.fftshift(
        make_frequency_grid((npts, npts, npts), voxel_size, half_space=False)
    )
    r = jnp.linalg.norm(xyz, axis=-1)
    r_fourier = r = jnp.linalg.norm(xyz_fourier, axis=-1)
    density_3d_true = jnp.exp(-(r**2) / (2 * sigma**2))
    fourier_density_3d_true = jnp.exp(-(r_fourier**2) / (2 * sigma_fourier**2))

    # Instantiate API
    # pose = cxs.EulerAnglePose(view_phi=phi, view_theta=theta, view_psi=psi)
    frequency_slice_in_pixels = make_frequency_slice((npts, npts), half_space=False)
    print(frequency_slice_in_pixels.shape)
    potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    fourier_slice_with_zero_in_corner = (
        potential_integrator.extract_voxels_from_grid_points(
            fourier_density_3d_true.astype(complex),
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms=convert_keV_to_angstroms(300.0),
        )
    )
    image_fourier = jnp.fft.fftshift(fourier_slice_with_zero_in_corner, axes=(0,))
    # image_real = irfftn(fourier_slice_with_zero_in_corner, s=(npts, npts))
    # print(image_fourier.shape, image_fourier.dtype)

    # plt.imshow(image_fourier.real)
    # plt.savefig("./plots/test_map_coordinates_1.png")

    xyz_slice = frequency_slice_in_pixels[0]
    # print(xyz_slice)
    # xyz_slice = xyz_slice.reshape(-1, 3)

    r_slice = np.sqrt(
        xyz_slice[:, :, 0] ** 2 + xyz_slice[:, :, 1] ** 2 + xyz_slice[:, :, 2] ** 2
    )
    slice_density_true = np.exp(-(r_slice**2) / (2 * sigma**2))

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

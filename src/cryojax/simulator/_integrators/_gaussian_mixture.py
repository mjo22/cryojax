"""
Scattering methods for the gaussian mixtures of atoms.
"""

'''
import jax.numpy as jnp

from .._potential._atom_potential import AtomCloud
from ._potential_integrator import AbstractPotentialIntegrator


class IndependentAtomScattering(AbstractPotentialIntegrator):
    """
    Projects a pointcloud of atoms onto the imaging plane.
    In contrast to the work in project_with_nufft, here each atom is

    TODO: Typehints for atom_density_kernel
    """

    def __call__(
        self,
        density: AtomCloud,
        # density: RealCloud,
        # coordinates: CloudCoords,
        # identity: IntCloud,
        # variances: IntCloud,  # WHAT SHOULD THE TYPE BE HERE?
        return_Fourier: bool = True,  # Michael: Conventionally I've been using "real"
        # for the fourier option (see Pose.rotate and ElectronDensity.real).
        # ... I suppose bools maybe should be called things like "is_real" though.
    ) -> ComplexImage:
        """
        Projects a pointcloud of atoms onto the imaging plane.
        In contrast to the work in project_with_nufft, here each atom is

        TODO: Typehints for atom_density_kernel
        """
        # Asserts may not jit compile. There is some experimental jax support
        # for something similar, but it's a pain right now. Exception handling
        # will work though because padded_shape is statically typed at compile
        # time.
        assert self.config.padded_shape[0] == self.config.padded_shape[1]
        pixel_grid = _build_pixel_grid(self.config.padded_shape[0], self.pixel_size)
        sq_distance = _evaluate_coord_to_grid_sq_distances(
            density.coordinates, pixel_grid
        )

        atom_variances = density.variances[density.identity]
        weights = density.weights[density.identity]
        gaussian_kernel = _eval_Gaussian_kernel(sq_distance, atom_variances) * weights
        print("after  egk")
        simulated_imgs = jnp.sum(gaussian_kernel, axis=-1)  # Sum over atoms
        if return_Fourier:
            # Michael: I try to standardize FFT conventions using
            # cryojax.utils.fft--using fftn method there would be best if
            # possible, but there are exceptions to this
            # (see line 106 of fourier_slice.extract_slice).
            simulated_imgs = jnp.fft.fft2(simulated_imgs)
        return simulated_imgs


# Michael: a few comments here---this is a situation where I would make use of utils.
# Maybe this should go in cryojax.utils.coordinates.py? Or somewhere else, haven't
# read in too much depth.
def _evaluate_coord_to_grid_sq_distances(
    x: PointCloudCoords2D, xgrid: ImageCoords
) -> ImageCoords:
    x_coords = jnp.expand_dims(x[:, :, 0], axis=1)  # N_struct x 1 x  N_atoms
    y_coords = jnp.expand_dims(x[:, :, 1], axis=1)
    x_sq_displacement = jnp.expand_dims((xgrid - x_coords) ** 2, axis=1)
    y_sq_displacement = jnp.expand_dims((xgrid - y_coords) ** 2, axis=2)
    # Todo: check that this is the image convention we want, and it shouldn't be 2, 1
    return x_sq_displacement + y_sq_displacement


def _eval_Gaussian_kernel(sq_distances, atom_variances) -> ImageCoords:
    print("inside egk")
    print(sq_distances.shape)
    print(atom_variances.shape)
    return jnp.exp(-sq_distances / (2 * atom_variances))


# Michael: similarly here. However in general, I try to avoid computing coordinate
# systems as much
# as possible. It can get very confusing having multiple coordinate systems around
# I think.
# To address this
# 1) For 2D coordinates I'm using ScatteringConfig.coords and ScatteringConfig.freqs
#    (also padded_coords and padded_freqs) and generally pass them as arguments. Could
#    these be used instead of this function?
#    Note: The exception to this is in Filters and Masks, where I do generate separate
#          but identical coordinate systems. To make sure no messiness happens, I use
#          the same functions from cryojax.utils.coordinates.
# 2) For 3D coordinates I also load from cryojax.utils.coordinates.
# Erik: Happy to change this, I just didn't know abouc cryojax.utils.coordinates could
# do this:
# The call to fftfreq mislead me into thinking it only did Fourier-space.
def _build_pixel_grid(
    npixels_per_side: int, pixel_size: RealNumber
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates the coordinates of each pixel in the image.  The center of the image
    is taken to be (0, 0).

    Args:
        npixels_per_side (float): Number of pixels on each side of the square miage
        pixel_size (int): Size of each pixel.

    Returns:
        tuple: two arrays containing the x, y coordinates of each pixel, respectively.
    """
    grid_1d = jnp.linspace(
        -npixels_per_side / 2, npixels_per_side / 2, npixels_per_side + 1
    )[:-1]
    grid_1d *= pixel_size
    return jnp.expand_dims(grid_1d, axis=(0, -1))
'''

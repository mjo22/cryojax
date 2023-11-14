"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = [
    "project_with_nufft",
    "extract_slice",
    "ImageConfig",
    "ScatteringConfig",
    "NufftScattering",
    "FourierSliceScattering",
]

from abc import abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core import field, Module
from ..types import (
    RealImage,
    ComplexImage,
    ImageCoords,
    ComplexVolume,
    VolumeCoords,
    RealCloud,
    IntCloud,
    CloudCoords,
)
from ..utils import (
    fftn,
    make_frequencies,
    make_coordinates,
    crop,
    pad,
    nufft,
    resize,
    map_coordinates,
)


class ImageConfig(Module):
    """
    Configuration for an electron microscopy image.

    Attributes
    ----------
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pad_scale :
        The scale at which to pad (or upsample) the image
        when computing it in the object plane. This
        should be a floating point number greater than
        or equal to 1. By default, it is 1 (no padding).
    freqs :
        The fourier wavevectors in the imaging plane.
    padded_freqs :
        The fourier wavevectors in the imaging plane
        in the padded coordinate system.
    coords :
        The coordinates in the imaging plane.
    padded_coords :
        The coordinates in the imaging plane
        in the padded coordinate system.
    """

    shape: tuple[int, int] = field(static=True)
    pad_scale: float = field(static=True, default=1.0)

    padded_shape: tuple[int, int] = field(static=True, init=False)

    freqs: ImageCoords = field(static=True, init=False)
    padded_freqs: ImageCoords = field(static=True, init=False)
    coords: ImageCoords = field(static=True, init=False)
    padded_coords: ImageCoords = field(static=True, init=False)

    def __post_init__(self):
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        self.padded_shape = padded_shape
        # Set coordinates
        self.freqs = make_frequencies(self.shape)
        self.padded_freqs = make_frequencies(self.padded_shape)
        self.coords = make_coordinates(self.shape)
        self.padded_coords = make_coordinates(self.padded_shape)

    def crop(self, image: RealImage) -> RealImage:
        """Crop an image."""
        return crop(image, self.shape)

    def pad(self, image: RealImage, **kwargs: Any) -> RealImage:
        """Pad an image."""
        return pad(image, self.padded_shape, **kwargs)

    def downsample(
        self, image: ComplexImage, method="lanczos5", **kwargs: Any
    ) -> ComplexImage:
        """Downsample an image."""
        return resize(
            image, self.shape, antialias=False, method=method, **kwargs
        )

    def upsample(
        self, image: ComplexImage, method="bicubic", **kwargs: Any
    ) -> ComplexImage:
        """Upsample an image."""
        return resize(image, self.padded_shape, method=method, **kwargs)


class ScatteringConfig(ImageConfig):
    """
    Configuration for an image with a particular
    scattering method.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.
    """

    @abstractmethod
    def scatter(self, *args: Any, **kwargs: Any) -> ComplexImage:
        """Scattering method for image rendering."""
        raise NotImplementedError


class FourierSliceScattering(ScatteringConfig):
    """
    Scatter points to the image plane using the
    Fourier-projection slice theorem.
    """

    order: int = field(static=True, default=1)
    mode: str = field(static=True, default="wrap")
    cval: complex = field(static=True, default=0.0 + 0.0j)

    def scatter(
        self,
        density: ComplexVolume,
        coordinates: VolumeCoords,
        resolution: float,
    ) -> ComplexImage:
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        return extract_slice(
            density,
            coordinates,
            resolution,
            self.padded_shape,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )


class NufftScattering(ScatteringConfig):
    """
    Scatter points to image plane using a
    non-uniform FFT.

    Attributes
    ----------
    eps : `float`
        See ``cryojax.utils.integration.nufft``
        for documentation.
    """

    eps: float = field(static=True, default=1e-6)

    def scatter(
        self, density: RealCloud, coordinates: CloudCoords, resolution: float
    ) -> ComplexImage:
        """Rasterize image with non-uniform FFTs."""
        return project_with_nufft(
            density,
            coordinates,
            resolution,
            self.padded_shape,
            eps=self.eps,
        )

class IndependentAtomScattering(ScatteringConfig):
    """
    Projects a pointcloud of atoms onto the imaging plane.
    In contrast to the work in project_with_nufft, here each atom is

    TODO: Typehints for atom_density_kernel
    """
    def scatter(
        self,
        density: RealCloud,
        coordinates: CloudCoords,
        resolution: float,
        identity: IntCloud,
        variances: IntCloud,  # WHAT SHOULD THE TYPE BE HERE?
        return_Fourier: bool = True,
    ) -> ComplexImage:
        """
        Projects a pointcloud of atoms onto the imaging plane.
        In contrast to the work in project_with_nufft, here each atom is

        TODO: Typehints for atom_density_kernel
        """
        assert(self.padded_shape[0] == self.padded_shape[1])
        pixel_grid = _build_pixel_grid(self.padded_shape[0], resolution)
        sq_distance = _evaluate_coord_to_grid_sq_distances(coordinates, pixel_grid)

        atom_variances = variances[identity]
        weights = density[identity]
        gaussian_kernel = _eval_Gaussian_kernel(sq_distance, atom_variances) * weights
        print("after  egk")
        simulated_imgs = jnp.sum(gaussian_kernel, axis=-1)  # Sum over atoms
        if return_Fourier:
            simulated_imgs = jnp.fft.fft2(simulated_imgs)
        return simulated_imgs


def _evaluate_coord_to_grid_sq_distances(
    x: CloudCoords, xgrid: ImageCoords
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


def _build_pixel_grid(
    npixels_per_side: int, pixel_size: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates the coordinates of each pixel in the image.  The center of the image  is taken to be (0, 0).

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
            

class IndependentAtomScatteringNufft(NufftScattering):
    """
    Projects a pointcloud of atoms onto the imaging plane.
    In contrast to the work in project_with_nufft, here each atom is

    TODO: Typehints for atom_density_kernel
    """

    def scatter(
        self,
        density: RealCloud,
        coordinates: CloudCoords,
        resolution: float,
        identity: IntCloud,
        atom_density_kernel,  # WHAT SHOULD THE TYPE BE HERE?
    ) -> ComplexImage:
        """
        Projects a pointcloud of atoms onto the imaging plane.
        In contrast to the work in project_with_nufft, here each atom is

        TODO: Typehints for atom_density_kernel
        """
        atom_types = jnp.unique(identity)

        img = jnp.zeros(self.padded_shape, dtype=jnp.complex64)
        for atom_type_i in atom_types:
            # Select the properties specific to that type of atom
            coords_i = coordinates[identity == atom_type_i]
            density_i = density[identity == atom_type_i]
            kernel_i = atom_density_kernel[atom_type_i]

            # Build an
            atom_i_image = project_with_nufft(
                density_i,
                coords_i,
                resolution,
                self.padded_shape,
                # atom_density_kernel[atom_type_i],
            )

            # img += atom_i_image * kernel_i
            img += atom_i_image 
        return img


def extract_slice(
    density: ComplexVolume,
    coordinates: VolumeCoords,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    density : shape `(N1, N2, N3)`
        Density grid in fourier space.
    coordinates : shape `(N1, N2, 1, 3)`
        Frequency central slice coordinate system.
    resolution :
        The rasterization resolution.
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    kwargs:
        Passed to ``cryojax.utils.interpolate.map_coordinates``.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    density, coordinates = jnp.asarray(density), jnp.asarray(coordinates)
    N1, N2, N3 = density.shape
    if not all([Ni == N1 for Ni in [N1, N2, N3]]):
        raise ValueError("Only cubic boxes are supported for fourier slice.")
    dx = resolution
    box_size = jnp.array([N1 * dx, N2 * dx, N3 * dx])
    # Need to convert to "array index coordinates".
    # Make coordinates dimensionless
    coordinates *= box_size
    # Interpolate on the upper half plane get the slice
    z = N2 // 2 + 1
    projection = map_coordinates(density, coordinates[:, :z], **kwargs)[..., 0]
    # Set zero frequency component to zero
    projection = projection.at[0, 0].set(0.0 + 0.0j)
    # Transform back to real space
    projection = jnp.fft.fftshift(jnp.fft.irfftn(projection, s=(N1, N2)))
    # Crop or pad to desired image size
    M1, M2 = shape
    if N1 >= M1 and N2 >= M2:
        projection = crop(projection, shape)
    elif N1 <= M1 and N2 <= M2:
        projection = pad(projection, shape, mode="edge")
    else:
        raise NotImplementedError(
            "density.shape must be larger or smaller than shape in all dimensions"
        )
    return fftn(projection) / jnp.sqrt(M1 * M2)


def project_with_nufft(
    density: RealCloud,
    coordinates: CloudCoords,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``cryojax.utils.integration.nufft`` for more detail.

    Arguments
    ---------
    density : shape `(N,)`
        Density point cloud.
    coordinates : shape `(N, 3)`
        Coordinate system of point cloud.
    resolution :
        The rasterization resolution.
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    kwargs:
        Passed to ``cryojax.utils.integration.nufft``.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    density, coordinates = jnp.asarray(density), jnp.asarray(coordinates)
    M1, M2 = shape
    image_size = jnp.array(np.array([M1, M2]) * resolution)
    coordinates = jnp.flip(coordinates[:, :2], axis=-1)
    projection = nufft(density, coordinates, image_size, shape, **kwargs)
    # Set zero frequency component to zero
    projection = projection.at[0, 0].set(0.0 + 0.0j)

    return projection / jnp.sqrt(M1 * M2)

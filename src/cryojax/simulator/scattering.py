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

from abc import ABCMeta, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core import (
    field,
    Module,
    RealImage,
    ComplexImage,
    ImageCoords,
    ComplexVolume,
    VolumeCoords,
    RealCloud,
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


class ScatteringConfig(ImageConfig, metaclass=ABCMeta):
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

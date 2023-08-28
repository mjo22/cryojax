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

from ..core import dataclass, field, Array, ArrayLike, CryojaxObject
from ..utils import (
    fft,
    make_frequencies,
    make_coordinates,
    crop,
    pad,
    nufft,
    resize,
    map_coordinates,
)


@dataclass
class ImageConfig(CryojaxObject):
    """
    Configuration for an electron microscopy image.

    Attributes
    ----------
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    freqs : `Array`, shape `(N1, N2, 2)`
        The fourier wavevectors in the imaging plane.
    padded_freqs : `Array`, shape `(M1, M2, 2)`
        The fourier wavevectors in the imaging plane
        in the padded coordinate system.
    coords : `Array`, shape `(N1, N2, 2)`
        The coordinates in the imaging plane.
    padded_coords : `Array`, shape `(M1, M2, 2)`
        The coordinates in the imaging plane
        in the padded coordinate system.
    pad_scale : `float`
        The scale at which to pad (or upsample) the image
        when computing it in the object plane. This
        should be a floating point number greater than
        or equal to 1. By default, it is 1 (no padding).
    """

    shape: tuple[int, int] = field(pytree_node=False, encode=tuple)

    padded_shape: tuple[int, int] = field(
        pytree_node=False, init=False, encode=False
    )

    freqs: Array = field(pytree_node=False, init=False, encode=False)
    padded_freqs: Array = field(pytree_node=False, init=False, encode=False)

    coords: Array = field(pytree_node=False, init=False, encode=False)
    padded_coords: Array = field(pytree_node=False, init=False, encode=False)

    pad_scale: float = field(pytree_node=False, default=1)

    def __post_init__(self):
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        object.__setattr__(self, "padded_shape", padded_shape)
        # Set coordinates
        freqs = make_frequencies(self.shape)
        padded_freqs = make_frequencies(self.padded_shape)
        coords = make_coordinates(self.shape)
        padded_coords = make_coordinates(self.padded_shape)
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "padded_freqs", padded_freqs)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "padded_coords", padded_coords)

    def crop(self, image: ArrayLike) -> Array:
        """Crop an image in real space."""
        return crop(image, self.shape)

    def pad(self, image: ArrayLike, **kwargs: Any) -> Array:
        """Pad an image in real space."""
        return pad(image, self.padded_shape, **kwargs)

    def downsample(
        self, image: ArrayLike, method="lanczos5", **kwargs: Any
    ) -> Array:
        """Downsample an image in Fourier space."""
        return resize(
            image, self.shape, antialias=False, method=method, **kwargs
        )

    def upsample(
        self, image: ArrayLike, method="bicubic", **kwargs: Any
    ) -> Array:
        """Upsample an image in Fourier space."""
        return resize(image, self.padded_shape, method=method, **kwargs)


@dataclass
class ScatteringConfig(ImageConfig, metaclass=ABCMeta):
    """
    Configuration for an image with a particular
    scattering method.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.
    """

    @abstractmethod
    def scatter(self, *args: Any, **kwargs: Any):
        """Scattering method for image rendering."""
        raise NotImplementedError


@dataclass
class FourierSliceScattering(ScatteringConfig):
    """
    Scatter points to the image plane using the
    Fourier-projection slice theorem.
    """

    order: int = field(pytree_node=False, default=1)
    mode: str = field(pytree_node=False, default="wrap")
    cval: complex = field(pytree_node=False, default=0.0 + 0.0j)

    def scatter(
        self, density: ArrayLike, coordinates: ArrayLike, resolution: float
    ) -> Array:
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


@dataclass
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

    eps: float = field(pytree_node=False, default=1e-6)

    def scatter(
        self, density: ArrayLike, coordinates: ArrayLike, resolution: float
    ) -> Array:
        """Rasterize image with non-uniform FFTs."""
        return project_with_nufft(
            density,
            coordinates,
            resolution,
            self.padded_shape,
            eps=self.eps,
        )


def extract_slice(
    density: ArrayLike,
    coordinates: ArrayLike,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    density : `ArrayLike`, shape `(N1, N2, N3)`
        Density grid in fourier space.
    coordinates : `ArrayLike`, shape `(N1, N2, 1, 3)`
        Frequency central slice coordinate system.
    resolution : float
        The rasterization resolution.
    shape : `tuple[int, int]`
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
    return fft(projection) / jnp.sqrt(M1 * M2)


def project_with_nufft(
    density: ArrayLike,
    coordinates: ArrayLike,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``cryojax.utils.integration.nufft`` for more detail.

    Arguments
    ---------
    density : `ArrayLike`, shape `(N,)`
        Density point cloud.
    coordinates : `ArrayLike`, shape `(N, 3)`
        Coordinate system of point cloud.
    resolution : float
        The rasterization resolution.
    shape : `tuple[int, int]`
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

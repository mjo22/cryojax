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

from ..core import dataclass, field, Array, CryojaxObject
from ..utils import (
    irfft,
    fft,
    fftfreqs,
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
    resolution : `float`
        Rasterization resolution.
        For voxel-based representations of ``Specimen``,
        this should be the same as the ``voxel_size``.
        This is in dimensions of length.
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
    resolution: float = field(pytree_node=False)

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
        freqs = jnp.asarray(fftfreqs(self.shape))
        padded_freqs = jnp.asarray(fftfreqs(self.padded_shape))
        coords = jnp.asarray(fftfreqs(self.shape, real=True))
        padded_coords = jnp.asarray(fftfreqs(self.padded_shape, real=True))
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "padded_freqs", padded_freqs)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "padded_coords", padded_coords)

    def crop(self, image: Array) -> Array:
        """Crop an image in real space."""
        return crop(image, self.shape)

    def pad(self, image: Array, **kwargs: Any) -> Array:
        """Pad an image in real space."""
        return pad(image, self.padded_shape, **kwargs)

    def downsample(
        self, image: Array, method="lanczos5", **kwargs: Any
    ) -> Array:
        """Downsample an image in Fourier space."""
        return resize(
            image, self.shape, antialias=False, method=method, **kwargs
        )

    def upsample(self, image: Array, method="bicubic", **kwargs: Any) -> Array:
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

    def scatter(self, *args):
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        return extract_slice(*args, self.padded_shape, order=self.order)


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

    def scatter(self, *args):
        """Rasterize image with non-uniform FFTs."""
        return project_with_nufft(*args, self.padded_shape, eps=self.eps)


def extract_slice(
    density: Array,
    coordinates: Array,
    voxel_size: Array,
    shape: tuple[int, int],
    **kwargs,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    density : `Array`, shape `(N1, N2, N3)`
        Density grid in fourier space.
    coordinates : `Array`, shape `(N1, N2, 1, 3)`
        Frequency central slice coordinate system.
    voxel_size : `Array`, shape `(3,)`
        Voxel size in each dimension.
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
    N1, N2, N3 = density.shape
    if not all([Ni == N1 for Ni in [N1, N2, N3]]):
        raise ValueError("Only cubic boxes are supported for fourier slice.")
    dx, dy, dz = voxel_size
    box_size = jnp.array([N1 * dx, N2 * dy, N3 * dz])
    # Need to convert to "array index coordinates".
    # Make coordinates dimensionless
    coordinates *= box_size
    # Interpolate and get the slice
    projection = map_coordinates(density, coordinates, **kwargs)[..., 0]
    # Set zero frequency component to zero
    projection = projection.at[0, 0].set(0.0 + 0.0j)
    # Crop or pad to desired image size
    projection = irfft(projection)
    M1, M2 = shape
    if N1 >= M1 and N2 >= M2:
        projection = crop(projection, shape)
    elif N1 <= M1 and N2 <= M2:
        projection = pad(projection, shape, mode="edge")
    else:
        raise NotImplementedError(
            "density.shape must be larger or smaller than shape in all dimensions"
        )
    return fft(projection)


def project_with_nufft(
    density: Array,
    coordinates: Array,
    voxel_size: Array,
    shape: tuple[int, int],
    eps: float = 1e-6,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``cryojax.utils.integration.nufft`` for more detail.

    Arguments
    ---------
    density : `Array`, shape `(N,)`
        Density point cloud.
    coordinates : `Array`, shape `(N, 3)`
        Coordinate system of point cloud.
    voxel_size : `Array`, shape `(3,)`
        Voxel size in each dimension.
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    eps : `float`
        Desired precision in computing the volume
        projection. See `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    N1, N2 = shape
    image_size = jnp.array(np.array([N1, N2]) * voxel_size[:2])
    coordinates = jnp.flip(coordinates[:, :2], axis=-1)
    projection = nufft(density, coordinates, image_size, shape, eps=eps)
    # Set zero frequency component to zero
    projection = projection.at[0, 0].set(0.0 + 0.0j)

    return projection

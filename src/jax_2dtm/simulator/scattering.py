"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = [
    "project_with_nufft",
    "project_with_gaussians",
    "project_with_slice",
    "ImageConfig",
    "ScatteringConfig",
    "NufftScattering",
    "GaussianScattering",
    "FourierSliceScattering",
]

from abc import ABCMeta, abstractmethod
from typing import Any

import jax.numpy as jnp
from jax.image import resize

from ..core import dataclass, field, Array, ArrayLike, Serializable
from ..utils import (
    fft,
    fftfreqs,
    bound,
    crop,
    pad,
    nufft,
    integrate_gaussians,
    resize,
    interpn,
)


@dataclass
class ImageConfig(Serializable):
    """
    Configuration for an electron microscopy image.

    Attributes
    ----------
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : `float`
        Size of camera pixels, in dimensions of length.
    freqs : Array, shape `(N1, N2, 2)`
        The fourier wavevectors in the imaging plane.
    padded_freqs : Array, shape `(M1, M2, 2)`
        The fourier wavevectors in the imaging plane
        in the padded coordinate system.
    coords : Array, shape `(N1, N2, 2)`
        The coordinates in the imaging plane.
    padded_coords : Array, shape `(M1, M2, 2)`
        The coordinates in the imaging plane
        in the padded coordinate system.
    """

    shape: tuple[int, int] = field(pytree_node=False, encode=tuple)
    pixel_size: float = field(pytree_node=False)

    padded_shape: tuple[int, int] = field(
        pytree_node=False, init=False, encode=False
    )

    freqs: ArrayLike = field(pytree_node=False, init=False, encode=False)
    padded_freqs: ArrayLike = field(
        pytree_node=False, init=False, encode=False
    )

    coords: ArrayLike = field(pytree_node=False, init=False, encode=False)
    padded_coords: ArrayLike = field(
        pytree_node=False, init=False, encode=False
    )

    pad_scale: float = field(pytree_node=False, default=1)

    def __post_init__(self):
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        object.__setattr__(self, "padded_shape", padded_shape)
        freqs = jnp.asarray(fftfreqs(self.shape, self.pixel_size))
        padded_freqs = jnp.asarray(
            fftfreqs(self.padded_shape, self.pixel_size)
        )
        coords = jnp.asarray(fftfreqs(self.shape, self.pixel_size, real=True))
        padded_coords = jnp.asarray(
            fftfreqs(self.padded_shape, self.pixel_size, real=True)
        )
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

    def downsample(self, image: Array, **kwargs: Any) -> Array:
        """Upsample an image in Fourier space."""
        return resize(image, self.shape, antialias=False, **kwargs)

    def upsample(self, image: Array, **kwargs: Any) -> Array:
        """Upsample an image in Fourier space."""
        return resize(image, self.padded_shape, **kwargs)


@dataclass
class ScatteringConfig(ImageConfig, metaclass=ABCMeta):
    """
    Configuration for an image with a given
    scattering method.
    """

    @abstractmethod
    def project(self, *args: Any):
        """Projection method for image rendering."""
        raise NotImplementedError


@dataclass
class FourierSliceScattering(ScatteringConfig):
    """
    Scatter points to the image plane using the
    Fourier-projection slice theorem.
    """

    def project(self, *args: Any, **kwargs: Any):
        """
        Compute image by interpolating onto the
        imaging plane.
        """
        density, coordinates, box_size = args
        projection = project_with_slice(
            density, coordinates, self.freqs, self.shape, **kwargs
        )
        return projection


@dataclass
class NufftScattering(ScatteringConfig):
    """
    Scatter points to image plane using a
    non-uniform FFT.

    Attributes
    ----------
    eps : `float`
        See ``jax_2dtm.utils.integration.nufft``
        for documentation.
    """

    eps: float = field(pytree_node=False, default=1e-6)

    def project(self, *args):
        """Rasterize image with non-uniform FFTs."""
        return project_with_nufft(*args, self.padded_shape, eps=self.eps)


@dataclass
class GaussianScattering(ScatteringConfig):
    """
    Scatter points to image by computing
    gaussian integrals.

    Attributes
    ----------
    scale : `float`
        Variance of a single gaussian density
        along each direction.
        See ``jax_2dtm.utils.integration.integrate_gaussians``
        for more documentation.
    """

    scale: float = field(pytree_node=False, default=1 / 3)

    def project(self, *args: Any):
        """Rasterize image by integrating over Gaussians."""
        return project_with_gaussians(
            *args,
            self.padded_shape,
            self.pixel_size,
            self.pixel_size * self.scale,
        )


def project_with_slice(
    density: Array,
    coordinates: Array,
    freqs: Array,
    shape: tuple[int, int],
    **kwargs,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    density :
        Density point cloud in fourier space.
    coordinates :
        Coordinate system of point cloud.
    freqs :
        The image coordinates over which to evaluate
        the interpolator.
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    kwargs:
        Passed to ``jax_2dtm.utils.interpolate.interpn``.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    N1, N2 = shape
    N = N1 * N2
    xi = jnp.append(freqs.reshape((N, 2)), jnp.zeros((N, 1)), axis=-1)
    flat = interpn(density, coordinates, xi)
    projection = flat.reshape(shape)
    return projection


def project_with_nufft(
    density: Array,
    coordinates: Array,
    box_size: Array,
    shape: tuple[int, int],
    eps: float = 1e-6,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``jax_2dtm.utils.integration.nufft`` for more detail.

    Arguments
    ---------
    density :
        Density point cloud.
    coordinates :
        Coordinate system of point cloud.
    box_size :
        Box size of points.
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
    masked = bound(density, coordinates[:, :2], box_size[:2])
    projection = nufft(
        masked, coordinates[:, :2], box_size[:2], shape, eps=eps
    )

    return projection


def project_with_gaussians(
    density: Array,
    coordinates: Array,
    box_size: Array,
    shape: tuple[int, int],
    pixel_size: float,
    scale: float,
) -> Array:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``jax_2dtm.utils.integration.integrate_gaussians`` for more detail.

    Arguments
    ---------
    density :
        Density point cloud.
    coordinates :
        Coordinate system of point cloud.
    box_size :
        Box size of points.
    shape : `tuple[int, int]`
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : `float`
        Pixel size
    scale : `float`
        Scale of gaussians in density point cloud.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    masked = bound(density, coordinates[:, :2], box_size[:2])
    projection = integrate_gaussians(
        masked,
        coordinates[:, :2],
        scale,
        shape,
        pixel_size,
    )

    return fft(projection)


def project_with_binning(
    density: Array, coords: Array, shape: tuple[int, int, int]
) -> Array:
    """
    Project 3D volume onto imaging plane
    using a histogram.

    Arguments
    ----------
    density : shape `(N,)`
        3D volume.
    coords : shape `(N, 3)`
        Coordinate system.
    shape :
        A tuple denoting the shape of the output image, given
        by ``(N1, N2)``
    Returns
    -------
    projection : shape `(N1, N2)`
        Projection of volume onto imaging plane,
        which is taken to be over axis 2.
    """
    N1, N2 = shape[0], shape[1]
    # Round coordinates for binning
    rounded_coords = jnp.rint(coords).astype(int)
    # Shift coordinates back to zero in the corner, rather than center
    x_coords, y_coords = (
        rounded_coords[:, 0] + N1 // 2,
        rounded_coords[:, 1] + N2 // 2,
    )
    # Bin values on the same y-z plane
    flat_coords = jnp.ravel_multi_index(
        (x_coords, y_coords), (N1, N2), mode="clip"
    )
    projection = jnp.bincount(
        flat_coords, weights=density, length=N1 * N2
    ).reshape((N1, N2))

    return projection

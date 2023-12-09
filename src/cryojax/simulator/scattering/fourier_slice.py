"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = ["extract_slice", "FourierSliceScattering"]

from typing import Any

import jax.numpy as jnp

from .base import ScatteringConfig
from ..density import VoxelGrid
from ...core import field
from ...types import (
    ComplexImage,
    ComplexVolume,
    VolumeCoords,
)
from ...utils import (
    fftn,
    crop,
    pad,
    map_coordinates,
)


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
        density: VoxelGrid,
        resolution: float,
    ) -> ComplexImage:
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        return extract_slice(
            density.weights,
            density.coordinates,
            resolution,
            self.padded_shape,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
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

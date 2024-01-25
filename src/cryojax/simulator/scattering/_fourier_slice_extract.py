"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = ["extract_slice", "FourierSliceExtract"]

from typing import Any

import jax.numpy as jnp

from ._scattering_model import ScatteringModel
from ..density import FourierVoxelGrid
from ...image import map_coordinates
from ...core import field
from ...typing import (
    ComplexImage,
    ComplexVolume,
    VolumeCoords,
)


class FourierSliceExtract(ScatteringModel):
    """
    Scatter points to the image plane using the
    Fourier-projection slice theorem.

    Attributes ``order``, ``mode``, and ``cval``
    are passed to ``cryojax.image.map_coordinates``.
    """

    interpolation_order: int = field(static=True, default=1)
    interpolation_mode: str = field(static=True, default="clip")
    interpolation_cval: complex = field(static=True, default=0.0 + 0.0j)

    def scatter(self, density: FourierVoxelGrid) -> ComplexImage:
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        return extract_slice(
            density.weights,
            density.frequency_slice.get(),
            order=self.interpolation_order,
            mode=self.interpolation_mode,
            cval=self.interpolation_cval,
        )


def extract_slice(
    weights: ComplexVolume,
    frequency_slice: VolumeCoords,
    order: int = 1,
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using the fourier slice theorem.

    Arguments
    ---------
    weights : shape `(N, N, N)`
        Density grid in fourier space. The zero frequency component
        should be in the center.
    frequency_slice : shape `(N, N, 1, 3)`
        Frequency central slice coordinate system, with the zero
        frequency component in the corner.
    order : int
        Spline order of interpolation. By default, ``1``.
    kwargs
        Keyword arguments passed to ``cryojax.image.map_coordinates``.

    Returns
    -------
    projection : shape `(N, N//2+1)`
        The output image in fourier space.
    """
    N1, N2, N3 = weights.shape
    N = N1
    if (N1, N2, N3) != (N, N, N):
        raise ValueError(
            "Only cubic boxes are supported for fourier slice theorem."
        )
    # Need to convert to logical coordinates, so first make coordinates dimensionless
    grid_shape = jnp.asarray([N, N, N], dtype=float)
    frequency_slice *= grid_shape
    # ... then shift from coordinates to indices
    frequency_slice += N // 2
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(frequency_slice, axes=[3, 0, 1, 2])
    projection = map_coordinates(weights, (k_x, k_y, k_z), order, **kwargs)[
        :, :, 0
    ]

    return jnp.fft.ifftshift(projection)[:, : N // 2 + 1]

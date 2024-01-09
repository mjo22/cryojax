"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = ["extract_slice", "FourierSliceExtract"]

from typing import Any

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from ._scattering_model import ScatteringModel
from ..density import VoxelGrid
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
    are passed to ``jax.scipy.map_coordinates``.
    """

    order: int = field(static=True, default=1)
    mode: str = field(static=True, default="wrap")
    cval: complex = field(static=True, default=0.0 + 0.0j)

    def scatter(self, density: VoxelGrid) -> ComplexImage:
        """
        Compute an image by sampling a slice in the
        rotated fourier transform and interpolating onto
        a uniform grid in the object plane.
        """
        return extract_slice(
            density.weights,
            density.frequency_slice,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
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
    weights : shape `(N1, N2, N3)`
        Density grid in fourier space.
    frequency_slice : shape `(N1, N2, 1, 3)`
        Frequency central slice coordinate system.
    order : int
        Spline order of interpolation. By default, ``1``.
    kwargs
        Keyword arguments passed to ``jax.scipy.ndimage.map_coordinates``.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    weights, frequency_slice = jnp.asarray(weights), jnp.asarray(
        frequency_slice
    )
    N1, N2, N3 = weights.shape
    if not all([Ni == N1 for Ni in [N1, N2, N3]]):
        raise ValueError("Only cubic boxes are supported for fourier slice.")
    # Need to convert to "array index coordinates", so make coordinates dimensionless
    box_size = jnp.array([N1, N2, N3], dtype=float)
    frequency_slice *= box_size
    # Transpose frequencies to map_coordinates convention
    frequency_slice = jnp.transpose(frequency_slice, axes=[3, 0, 1, 2])
    # Flip negative valued frequencies to get the logical coordinates.
    frequency_slice = jnp.where(
        frequency_slice < 0,
        box_size[:, None, None, None] + frequency_slice,
        frequency_slice,
    )
    return map_coordinates(weights, frequency_slice, order, **kwargs)[..., 0]

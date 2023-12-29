"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = ["extract_slice", "FourierSliceExtract"]

from typing import Any

import jax.numpy as jnp

from ._scattering_model import ScatteringModel
from ..density import VoxelGrid
from ...core import field
from ...typing import (
    ComplexImage,
    ComplexVolume,
    VolumeCoords,
)
from ...utils import map_coordinates


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
    kwargs:
        Passed to ``cryojax.utils.interpolate.map_coordinates``.

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
    # Interpolate on the upper half plane get the slice
    # z = N2 // 2 + 1
    # fourier_projection = map_coordinates(
    #    weights, coordinates[:, :z], **kwargs
    # )[..., 0]
    # Transform back to real space
    # fourier_projection = irfftn(fourier_projection, s=(N1, N2))
    #

    return map_coordinates(weights, frequency_slice, **kwargs)[..., 0]

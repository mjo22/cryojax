"""
Scattering methods for the fourier slice theorem.
"""

from __future__ import annotations

__all__ = ["extract_slice", "FourierSliceScattering"]

from typing import Any

import jax.numpy as jnp

from ._scattering import ScatteringConfig
from ..density import VoxelGrid
from ...core import field
from ..pose import Pose, EulerPose
from ...typing import (
    ComplexImage,
    ComplexVolume,
    VolumeCoords,
)
from ...utils import (
    fftn,
    irfftn,
    crop,
    pad,
    map_coordinates,
    make_coordinates,
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
    weights: ComplexVolume,
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
    weights : shape `(N1, N2, N3)`
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
    weights, coordinates = jnp.asarray(weights), jnp.asarray(coordinates)
    N1, N2, N3 = weights.shape
    if not all([Ni == N1 for Ni in [N1, N2, N3]]):
        raise ValueError("Only cubic boxes are supported for fourier slice.")
    dx = resolution
    box_size = jnp.array([N1 * dx, N2 * dx, N3 * dx])
    # Need to convert to "array index coordinates".
    # Make coordinates dimensionless
    coordinates *= box_size
    # Interpolate on the upper half plane get the slice
    z = N2 // 2 + 1
    projection = map_coordinates(weights, coordinates[:, :z], **kwargs)[..., 0]
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
            "Voxel density shape must be larger or smaller than shape in all dimensions"
        )
    return fftn(projection) / jnp.sqrt(M1 * M2)


def insert_slice_and_interpolate(slice_real, xyz_rotated_single, n_pix):
    from cryojax.utils import interpolate
    r0, r1, dd = interpolate.diff(xyz_rotated_single)
    map_3d_interp_slice, count_3d_interp_slice = interpolate.interp_vec(
        slice_real, r0, r1, dd, n_pix
    )
    inserted_slice_3d = map_3d_interp_slice.reshape((n_pix, n_pix, n_pix))
    count_3d = count_3d_interp_slice.reshape((n_pix, n_pix, n_pix))
    return inserted_slice_3d, count_3d


def insert_slice(
        slice_f: ComplexImage,
        pose: Pose = field(default_factory=EulerPose),
        to_real: bool = False,
) -> ComplexVolume:
    """
    Insert a slice into a 3D volume using the fourier slice theorem.

    Arguments
    ---------
    slice : shape `(M1, M2)`
        The slice to insert into the volume.

    Returns
    -------
    volume : shape `(N,N,N)`
        The volume (in Fourier space) with the slice inserted.
    """
    M1, M2 = slice_f.shape
    assert M1 == M2, "Slice must be square"
    N = M1
    
    xyz = jnp.zeros((3,N**2))
    rotation = pose.rotation.as_matrix()
    xyz_central_slice = xyz.at[:2].set(make_coordinates((N,N)).reshape(-1,2).T)
    xyz_rotated_central_slice = rotation @ xyz_central_slice

    inserted_slice_3d_real, count_3d_real = insert_slice_and_interpolate(slice_f.real, xyz_rotated_central_slice, N)
    inserted_slice_3d_imag, count_3d_imag = insert_slice_and_interpolate(slice_f.imag, xyz_rotated_central_slice, N)

    volume = inserted_slice_3d_real*count_3d_real + 1j*inserted_slice_3d_imag*count_3d_imag

    if to_real:
        volume = irfftn(jnp.fft.ifftshift(volume))

    return volume
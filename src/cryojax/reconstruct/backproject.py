"""
Reconstruction methods for backprojection (the fourier slice theorem).
"""


from __future__ import annotations

__all__ = ["WeinerFilter", "insert_slice", "_insert_slice_and_interpolate"]

from cryojax.simulator import Filter
from cryojax.typing import RealImage

from typing import Any

import jax.numpy as jnp

from ..core import field
from ..simulator.pose import Pose, EulerPose
from typing import Union
from ..typing import (
    ComplexImage,
    ComplexVolume,
    Volume,
)
from ..utils import (
    ifftn,
    make_coordinates,
    make_frequencies,
)
from cryojax.utils import interpolate


class WeinerFilter(Filter):
    def __init__(self, ctf: RealImage, noise_level: float = 0.0):
        self.filter = ctf / (ctf * ctf + noise_level)


def _insert_slice_and_interpolate(slice_real, xyz_rotated_single, n_pix):
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
) -> Union[Volume, ComplexVolume]:
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

    xyz = jnp.zeros((3, N**2))
    rotation = pose.rotation.as_matrix()
    xyz_central_slice = xyz.at[:2].set(
        make_coordinates((N, N)).reshape(-1, 2).T
    )
    xyz_rotated_central_slice = rotation @ xyz_central_slice

    inserted_slice_3d_real, count_3d_real = _insert_slice_and_interpolate(
        slice_f.real, xyz_rotated_central_slice, N
    )
    inserted_slice_3d_imag, count_3d_imag = _insert_slice_and_interpolate(
        slice_f.imag, xyz_rotated_central_slice, N
    )

    volume = (
        inserted_slice_3d_real * count_3d_real
        + 1j * inserted_slice_3d_imag * count_3d_imag
    )

    if to_real:
        volume = ifftn(jnp.fft.ifftshift(volume)).real

    return volume


def filtered_backprojection(
    deconolved_images_f: [ComplexImage, ...],
    poses: [Pose, ...],
    to_real: bool = True,
) -> Union[Volume, ComplexVolume]:
    """
    Performs filtered backprojection reconstruction on a stack of deconvolved images.

    Args:
        deconolved_images (ndarray): Stack of deconvolved images.
        poses (ndarray): Array of poses corresponding to each deconvolved image.

    Returns:
        ndarray: Reconstructed volume.

    """
    n_slices = len(deconolved_images_f)
    n_pix = deconolved_images_f.shape[-1]
    volume_sum_f = jnp.zeros((n_pix, n_pix, n_pix), dtype=jnp.complex64)
    freq_xy = jnp.fft.fftshift(
        make_frequencies((n_pix, n_pix), half_space=False)
    )
    freq_abs = jnp.hypot(freq_xy[:, :, 0], freq_xy[:, :, 1])
    for idx in range(n_slices):
        image_deconv = deconolved_images_f[idx]
        pose = poses[idx]
        slice_f_ram_lak_filtered = freq_abs * image_deconv
        volume_f = insert_slice(slice_f_ram_lak_filtered, pose, to_real=False)
        volume_sum_f += volume_f
    volume_f = volume_sum_f / n_slices
    if to_real:
        volume_r = ifftn(jnp.fft.ifftshift(volume_sum_f)).real * jnp.sqrt(
            n_pix**3
        )
        return volume_r
    else:
        return volume_f

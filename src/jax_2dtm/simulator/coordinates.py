"""
Routines defining coordinate system for 3D templates or images.
"""

__all__ = ["coordinatize", "radial_grid"]

import jax.numpy as jnp
import numpy as np
from typing import Optional, Sequence, Tuple
from jax_2dtm.types import Array, ArrayLike


def coordinatize(
    volume: ArrayLike, pixel_size: float, eps: Optional[float] = None
) -> Tuple[Array, Array]:
    """
    Returns flattened coordinate system and 3D volume or 2D image
    of shape ``(N, ndim)``, where ``ndim = volume.ndim`` and
    ``N = N1*N2*N3 - M`` or ``N = N2*N3 - M``, where ``M`` is a
    number of points close to zero that are masked out.
    The coordinate system is in pixel coordinates with zero
    in the center.

    Parameters
    ----------
    volume : shape `(N1, N2, N3)` or `(N2, N3)`
        3D volume or 2D image.
    pixel_size : optional
        Camera pixel size.
    eps : optional
        Remove points from the volume where the
        density is below this threshold.

    Returns
    -------
    cloud : shape `(N, ndim)`
        Flattened volume or image.
    coords : shape `(N, ndim)`
        Flattened cartesian coordinate system.
    """
    ndim, shape = volume.ndim, volume.shape
    if eps is None:
        eps = float(np.finfo(volume.dtype).eps)

    # Mask out points where the electron density below threshold
    flat = volume.ravel()
    mask = np.where(flat > eps)
    cloud = flat[mask]

    # Create coordinate buffer
    N = cloud.size
    coords = np.zeros((N, ndim))

    # Generate cubic grid and fill coordinate array
    R = radial_grid(shape)
    for i in range(ndim):
        coords[:, i] = pixel_size * R[i].ravel()[mask]

    return jnp.asarray(cloud), jnp.asarray(coords)


def radial_grid(shape: tuple[int, ...]) -> tuple[ArrayLike, ...]:
    """
    Create a radial coordinate system on a grid.
    This can be used for real and fourier space
    calculations. If used for fourier space, the
    zero-frequency component is in the center.

    Arguments
    ---------
    shape :
        Shape of the voxel grid. Can be 2D or 3D.

    Returns
    -------
    rcoords :
        2D or 3D cartesian coordinate system with
        zero in the center.
    """
    ndim = len(shape)
    rcoords1D = []
    for i in range(ndim):
        ni = shape[i]
        ri = np.fft.fftshift(jnp.fft.fftfreq(ni)) * ni
        rcoords1D.append(ri)

    rcoords = np.meshgrid(*rcoords1D, indexing="ij")

    return rcoords

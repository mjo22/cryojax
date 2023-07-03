"""
Routines defining coordinate system for 3D templates or images.
"""

__all__ = ["radial_grid", "coordinatize"]

import jax.numpy as jnp
import numpy as np
from typing import Optional
from jax_2dtm.types import Array, ArrayLike


def coordinatize(
    template: ArrayLike, pixel_size: float, threshold: Optional[float] = None
) -> tuple[Array, ...]:
    """
    Returns flattened coordinate system and 3D volume or 2D image
    of shape ``(N, ndim)``, where ``ndim = volume.ndim`` and
    ``N = N1*N2*N3 - M`` or ``N = N2*N3 - M``, where ``M`` is a
    number of points close to zero that are masked out.
    The coordinate system is in pixel coordinates with zero
    in the center.

    Parameters
    ----------
    density : shape `(N1, N2, N3)` or `(N1, N2)`
        3D volume or 2D image on a cartesian grid.
    pixel_size : float
        Camera pixel size.
    threshold : float, optional
        Remove points from the volume where the
        density is below this threshold.

    Returns
    -------
    density : shape `(N, ndim)`
        Point cloud volume or image.
    coords : shape `(N, ndim)`
        Point cloud cartesian coordinate system.
    """
    ndim, shape = template.ndim, template.shape
    if threshold is None:
        threshold = float(np.finfo(template.dtype).eps)

    # Mask out points where the electron density below threshold
    flat = template.ravel()
    mask = np.where(flat > threshold)
    density = flat[mask]

    # Create coordinate buffer
    N = density.size
    coords = np.zeros((N, ndim))

    # Generate cubic grid and fill coordinate array
    R = radial_grid(shape)
    for i in range(ndim):
        coords[:, i] = pixel_size * R[i].ravel()[mask]

    return jnp.array(density), jnp.array(coords)


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

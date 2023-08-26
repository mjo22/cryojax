"""
Integration routines.
"""

__all__ = ["nufft", "integrate_gaussians"]

from typing import Any, Union

import jax
import jax.numpy as jnp
import tensorflow_nufft as tfft
from jax.experimental import jax2tf

# from jax_finufft import nufft1
from jax.scipy import special

from .coordinates import fftfreqs1d
from ..core import Array, ArrayLike


def nufft(
    density: ArrayLike,
    coords: ArrayLike,
    box_size: Union[float, ArrayLike],
    shape: tuple[int, int],
    eps: float = 1e-6,
    **kwargs: Any
) -> Array:
    r"""
    Helper routine to compute a non-uniform FFT on the
    imaging plane for a 3D point cloud.
    Mask out points that lie out of bounds.

    .. warning::
        If any values in ``coords`` lies out of bounds of
        :math:`$(-3\pi, 3\pi]$`, this method will crash.
        This means that ``density`` cannot be
        arbitrarily cropped, only to a certain extent.

    Arguments
    ---------
    density : `ArrayLike`, shape `(N,)`
        Density point cloud over which to compute
        the fourier transform.
    coords : `ArrayLike`, shape `(N, 2)`
        Coordinate system for density cloud.
    box_size : `float` or `ArrayLike`, shape `(2,)`
        2D imaging plane that ``coords`` lies in.
    shape : `tuple[int, int]`
        Desired output shape of the transform.
    eps : `float`
        Precision of the non-uniform FFT. See
        `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    Return
    ------
    ft : `Array`, shape ``shape``
        Fourier transform.
    """
    coords = jnp.asarray(coords)
    complex_density = jnp.asarray(density.astype(complex))
    periodic_coords = 2 * jnp.pi * coords / box_size

    def tf_nufft1(shape, density, coords, **kwargs):
        return tfft.nufft(
            density,
            coords,
            grid_shape=shape,
            transform_type="type_1",
            tol=eps,
            **kwargs
        )

    nufft1 = jax2tf.call_tf(
        tf_nufft1,
        output_shape_dtype=jax.ShapeDtypeStruct(shape, complex_density.dtype),
    )
    # ft = nufft1(
    #    shape, complex_density, jnp.flip(periodic_coords, axis=-1), **kwargs
    # )
    ft = nufft1(shape, complex_density, periodic_coords, **kwargs)
    # x, y = periodic_coords.T
    # ft = nufft1(shape, complex_density, -y, -x, eps=eps)

    return jnp.fft.ifftshift(ft)


def integrate_gaussians(
    weights: ArrayLike,
    centers: ArrayLike,
    scales: ArrayLike,
    shape: tuple[int, int],
    pixel_size: float,
) -> Array:
    """
    Integrate a sum of Gaussians over a grid given by the given shape.

    Computes the integral of a function which is represented as a weighted sum of Gaussians
    with the given centers and scales. The coordinate system corresponds to the grid with
    unit spacing and with the origin at the center of the grid.

    Parameters
    ----------
    shape : `tuple[int, int]`
        Shape of grid to integrate over.
        The number of dimensions is inferred from the length of this sequence.
    pixel_size : `float`
        Pixel size.
    weights : `ArrayLike`, shape `(N,)`
        Weights of Gaussian densities.
    centers : `ArrayLike`, shape `(N, 2)`
        Centers of Gaussian densities.
    scales : `ArrayLike`, shape `(N,)`
        Scales of Gaussian densities.

    Returns
    -------
    image : `Array`, shape `shape`
        Integrals of Gaussian densities over grid.
    """
    x, y = [fftfreqs1d(s + 1, pixel_size, real=True) for s in shape]
    image = _integrate_gaussians(x, y, weights, centers, scales)

    return image


@jax.jit
def _integrate_gaussians(
    x: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike,
    centers: ArrayLike,
    scales: ArrayLike,
) -> Array:
    """
    Integrate a Gaussian density over a set of intervals given by boundaries.

    Parameters
    ----------
    x : `ArrayLike`, shape `(M1 + 1,)`
        x boundary of intervals to integrate over.
    y : `ArrayLike`, shape `(M2 + 1,)`
        y boundary of intervals to integrate over.
    weights : `ArrayLike`, shape `(N,)`
        Gaussian weights
    centers : `ArrayLike`, shape `(N, 2)`
        Centers of Gaussian densities.
    scales : `ArrayLike`, shape `(N,)`
        Scales of Gaussian densities.

    Returns
    -------
    I : `Array`, shape `(M1, M2)`
        Integrals of Gaussian densities over intervals.
    """
    centers = jnp.expand_dims(centers, axis=-1)
    scales = jnp.expand_dims(scales, axis=-1)

    # Marginals
    I_x = 0.5 * jnp.diff(
        special.erf((x - centers[:, 0]) / (scales * jnp.sqrt(2.0))), axis=-1
    )
    I_y = 0.5 * jnp.diff(
        special.erf((y - centers[:, 1]) / (scales * jnp.sqrt(2.0))), axis=-1
    )
    # Contract by summing over batch dimension and outer product over spatial dimensions
    # In summation notation, this would correspond to the einsum
    # jnp.einsum('mi,mj,m->ij', *marginals, weights)
    I = jnp.matmul((I_x * jnp.expand_dims(weights, axis=-1)).T, I_y)

    return I

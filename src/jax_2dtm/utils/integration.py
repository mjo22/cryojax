"""
Integration routines.
"""

__all__ = ["nufft", "integrate_gaussians"]

import jax.numpy as jnp

# import tensorflow as tf
# import tensorflow_nufft as tfft
# from jax.experimental import jax2tf
from jax_finufft import nufft1
from jax.scipy import special

from .fft import fftfreqs1d
from ..core import Array


def nufft(
    density: Array,
    coords: Array,
    box_size: Array,
    shape: tuple[int, int],
    eps: float = 1e-6,
) -> Array:
    r"""
    Helper routine to compute a non-uniform FFT on the
    imaging plane for a 3D point cloud.
    Mask out points that lie out of bounds.

    .. warning::
        If any values in ``coords`` lies out of bounds of
        :math:`$(-3\pi, 3\pi]$`, this method will crash.
        This means that ``density`` cannot be
        arbitrarily cropped and translated out of frame,
        rather only to a certain extent.

    Arguments
    ---------
    density : shape `(N,)`
        Density point cloud over which to compute
        the fourier transform.
    coords : shape `(N, 2)`
        Coordinate system for density cloud.
    box_size : shape `(2,)`
        2D imaging plane that ``coords`` lies in.
    shape :
        Desired output shape of the transform.
    eps :
        Precision of the non-uniform FFT. See
        `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    Return
    ------
    ft : Array, shape ``shape``
        Fourier transform.
    """
    complex_density = density.astype(complex)
    periodic_coords = 2 * jnp.pi * coords / box_size
    # _nufft1 = jax2tf.call_tf(_tf_nufft1, output_shape_dtype=jax.ShapeDtypeStruct(shape, masked_density.dtype))
    # ft = _nufft1(masked_density, periodic_coords, shape, eps)
    x, y = periodic_coords.T
    # ft = nufft1(shape, complex_density, y, x, eps=eps)[::-1, ::-1]
    ft = nufft1(shape, complex_density, -y, -x, eps=eps)

    return ft


def integrate_gaussians(
    weights: Array,
    centers: Array,
    scales: Array,
    shape: tuple[int, int],
    pixel_size: float,
):
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
    weights : `Array`, shape `(N,)`
        Weights of Gaussian densities.
    centers : `Array`, shape `(N, 2)`
        Centers of Gaussian densities.
    scales : `Array`, shape `(N,)`
        Scales of Gaussian densities.

    Returns
    -------
    image : `Array`, shape `shape`
        Integrals of Gaussian densities over grid.
    """

    # Compute marginals in x and y
    marginals = [
        _integrate_gaussian_1d(
            fftfreqs1d(s + 1, pixel_size, real=True),
            centers[:, i],
            scales,
        )
        for i, s in enumerate(shape)
    ]

    # Contract by summing over batch dimension and outer product over spatial dimensions
    # In summation notation, this would correspond to the einsum
    # jnp.einsum('mi,mj,m->ij', *marginals, weights)
    my, mx = marginals
    image = jnp.matmul((mx * jnp.expand_dims(weights, axis=-1)).T, my)

    return image


# def _tf_nufft1(source, points, shape, tol):
#    """
#    Wrapper for type-1 non-uniform FFT
#    from tensorflow-nufft.
#    """
#    return tfft.nufft(
#        source,
#        points,
#        grid_shape=shape,
#        transform_type="type_1",
#        tol=tol.numpy(),
#    )


def _integrate_gaussian_1d(boundaries: Array, centers: Array, scales: Array):
    """
    Integrate a Gaussian density over a set of intervals given by boundaries.

    Parameters
    ----------
    boundaries : `Array`, shape `(M + 1,)`
        Boundaries of intervals to integrate over.
    centers : `Array`, shape `(N,)`
        Centers of Gaussian densities.
    scales : `Array`, shape `(N,)`
        Scales of Gaussian densities.

    Returns
    -------
    I : `Array`, shape `(N, M)`
        Integrals of Gaussian densities over intervals.
    """
    centers = jnp.expand_dims(centers, axis=-1)
    scales = jnp.expand_dims(scales, axis=-1)

    z = (boundaries - centers) / (scales * jnp.sqrt(2.0))
    I = 0.5 * jnp.diff(special.erf(z), axis=-1)

    return I

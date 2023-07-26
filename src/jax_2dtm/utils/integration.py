"""
Integration routines.
"""

__all__ = ["nufft", "integrate_gaussians"]

import jax
import jax.numpy as jnp
#import tensorflow_nufft as tfft
#from jax.experimental import jax2tf

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
    #nufft1 = jax2tf.call_tf(
    #    _tf_nufft1,
    #    output_shape_dtype=jax.ShapeDtypeStruct(shape, complex_density.dtype),
    #)
    #ft = nufft1(
    #    complex_density, jnp.flip(periodic_coords, axis=-1), shape, eps
    #)
    x, y = periodic_coords.T
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
    x, y = [fftfreqs1d(s + 1, pixel_size, real=True) for s in shape]
    image = _integrate_gaussians(x, y, weights, centers, scales)

    return image


#def _tf_nufft1(source, points, shape, tol):
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


@jax.jit
def _integrate_gaussians(
    x: Array, y: Array, weights: Array, centers: Array, scales: Array
):
    """
    Integrate a Gaussian density over a set of intervals given by boundaries.

    Parameters
    ----------
    x : `Array`, shape `(M1 + 1,)`
        x boundary of intervals to integrate over.
    y : `Array`, shape `(M2 + 1,)`
        y boundary of intervals to integrate over.
    weights : `Array`, shape `(N,)`
        Gaussian weights
    centers : `Array`, shape `(N, 2)`
        Centers of Gaussian densities.
    scales : `Array`, shape `(N,)`
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

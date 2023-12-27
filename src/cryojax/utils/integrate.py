"""
Integration routines.
"""

__all__ = ["nufft", "integrate_gaussians"]

from typing import Any, Union
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
from jax.scipy import special

from .coordinates import _fftfreqs1d

from ..typing import (
    RealVector,
    RealCloud,
    CloudCoords2D,
    ComplexImage,
    RealImage,
)


def nufft(
    density: RealCloud,
    coords: CloudCoords2D,
    box_size: Union[Float[Array, ""], Float[Array, "2"]],
    shape: tuple[int, int],
    eps: float = 1e-6,
    **kwargs: Any
) -> ComplexImage:
    r"""
    Helper routine to compute a non-uniform FFT on the
    imaging plane for a 3D point cloud.

    .. warning::
        If any values in ``2 * np.pi * coords / box_size``
        lies out of bounds of :math:`$(-3\pi, 3\pi]$`,
        this method will crash.

    Arguments
    ---------
    density :
        Density point cloud over which to compute
        the fourier transform.
    coords :
        Coordinate system for density cloud.
    box_size :
        2D imaging plane that ``coords`` lies in.
    shape :
        Desired output shape of the transform.
    eps :
        Precision of the non-uniform FFT. See
        `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    kwargs :
        Keyword arguments passed to ``jax_finufft.nufft1``.
    Return
    ------
    ft : Fourier transform.
    """
    from jax_finufft import nufft1

    coords = jnp.asarray(coords)
    complex_density = jnp.asarray(density.astype(complex))
    periodic_coords = 2 * jnp.pi * coords / box_size

    x, y = periodic_coords.T
    ft = nufft1(shape, complex_density, -x, -y, eps=eps, **kwargs)

    return jnp.fft.ifftshift(ft)


def integrate_gaussians(
    weights: RealCloud,
    centers: CloudCoords2D,
    variances: RealCloud,
    shape: tuple[int, int],
    pixel_size: float,
) -> RealImage:
    """
    Integrate a sum of Gaussians over a grid given by the given shape.

    Computes the integral of a function which is represented as a weighted sum of Gaussians
    with the given centers and scales. The coordinate system corresponds to the grid with
    unit spacing and with the origin at the center of the grid.

    Parameters
    ----------
    weights :
        Weights of Gaussian densities.
    centers :
        Centers of Gaussian densities.
    variances :
        Variances of Gaussian densities.
    shape :
        Shape of grid to integrate over.
        The number of dimensions is inferred from the length of this sequence.
    pixel_size :
        Pixel size.

    Returns
    -------
    image : `Array`, shape `shape`
        Integrals of Gaussian densities over grid.
    """
    x, y = [_fftfreqs1d(s + 1, pixel_size, real=True) for s in shape]
    image = _integrate_gaussians(x, y, weights, centers, variances)

    return image


@jax.jit
def _integrate_gaussians(
    x: RealVector,
    y: RealVector,
    weights: RealCloud,
    centers: RealCloud,
    scales: RealCloud,
) -> RealImage:
    """
    Integrate a Gaussian density over a set of intervals given by boundaries.

    Parameters
    ----------
    x : shape `(M1 + 1,)`
        x boundary of intervals to integrate over.
    y : shape `(M2 + 1,)`
        y boundary of intervals to integrate over.
    weights :
        Gaussian weights
    centers :
        Centers of Gaussian densities.
    scales :
        Scales of Gaussian densities.

    Returns
    -------
    I : shape `(M1, M2)`
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

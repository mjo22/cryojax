"""
Routines to compute FFTs, in cryojax conventions.
"""

from typing import Any, Optional, overload

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact


@overload
def ifftn(
    ft: Complex[Array, "Nz Ny Nx"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Nz Ny Nx"]: ...


@overload
def ifftn(
    ft: Complex[Array, "Ny Nx"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx"]: ...


def ifftn(
    ft: Complex[Array, "Ny Nx"] | Complex[Array, "Nz Ny Nx"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx"] | Complex[Array, "Nz Ny Nx"]:
    """
    Helper routine to match the inverse fourier transform
    to the output of a type 1 non-uniform FFT with jax-finufft.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the center.
    real : `bool`
        If ``True``, then ``ft`` is the Fourier transform
        of a real-valued function.
    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    ift = jnp.fft.fftshift(jnp.fft.ifftn(ft, axes=axes, **kwargs), axes=axes)

    if real:
        return ift.real
    else:
        return ift


@overload
def fftn(
    ift: Inexact[Array, "Nz Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Nz Ny Nx"]: ...


@overload
def fftn(
    ift: Inexact[Array, "Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx"]: ...


def fftn(
    ift: Inexact[Array, "Ny Nx"] | Inexact[Array, "Nz Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx"] | Complex[Array, "Nz Ny Nx"]:
    """
    Helper routine to match the fourier transform of an array
    with the output of a type 1 non-uniform FFT with jax-finufft.

    Arguments
    ---------
    ift :
        Array in real space. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ft :
        Fourier transform of array.
    """
    ft = jnp.fft.fftn(jnp.fft.ifftshift(ift, axes=axes), axes=axes, **kwargs)

    return ft


@overload
def irfftn(
    ft: Complex[Array, "Ny Nx//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Float[Array, "Ny Nx"] | Float[Array, " *s"]: ...


@overload
def irfftn(
    ft: Complex[Array, "Nz Ny Nx//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Float[Array, "Nz Ny Nx"] | Float[Array, " *s"]: ...


def irfftn(
    ft: Complex[Array, "Ny Nx//2+1"] | Complex[Array, "Nz Ny Nx//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Float[Array, "Ny Nx"] | Float[Array, "Nz Ny Nx"] | Float[Array, " *s"]:
    """
    Helper routine to compute the inverse fourier transform
    from real input.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    ift = jnp.fft.fftshift(jnp.fft.irfftn(ft, s=s, axes=axes, **kwargs), axes=axes)

    return ift


@overload
def rfftn(
    ift: Float[Array, "Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx//2+1"]: ...


@overload
def rfftn(
    ift: Float[Array, "Nz Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Nz Ny Nx//2+1"]: ...


def rfftn(
    ift: Float[Array, "Ny Nx"] | Float[Array, "Nz Ny Nx"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "Ny Nx//2+1"] | Complex[Array, "Nz Ny Nx//2+1"]:
    """
    Helper routine to compute the fourier transform of an array
    to match the output of a type 1 non-uniform FFT.

    Arguments
    ---------
    ift :
        Array in real space. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ft :
        Fourier transform of array.
    """
    ft = jnp.fft.rfftn(jnp.fft.ifftshift(ift, axes=axes), axes=axes, **kwargs)

    return ft

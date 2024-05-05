"""
Routines to compute FFTs, in cryojax conventions.
"""

from typing import Any, Optional, overload

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact


@overload
def ifftn(
    ft: Complex[Array, "z_dim y_dim x_dim"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "z_dim y_dim x_dim"]: ...


@overload
def ifftn(
    ft: Complex[Array, "y_dim x_dim"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim"]: ...


def ifftn(
    ft: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]:
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
    ift: Inexact[Array, "z_dim y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "z_dim y_dim x_dim"]: ...


@overload
def fftn(
    ift: Inexact[Array, "y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim"]: ...


def fftn(
    ift: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]:
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
    ft: Complex[Array, "y_dim x_dim//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Float[Array, "y_dim x_dim"] | Float[Array, " *s"]: ...


@overload
def irfftn(
    ft: Complex[Array, "z_dim y_dim x_dim//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Float[Array, "z_dim y_dim x_dim"] | Float[Array, " *s"]: ...


def irfftn(
    ft: Complex[Array, "y_dim x_dim//2+1"] | Complex[Array, "z_dim y_dim x_dim//2+1"],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> (
    Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"] | Float[Array, " *s"]
):
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
    ift: Float[Array, "y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim//2+1"]: ...


@overload
def rfftn(
    ift: Float[Array, "z_dim y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "z_dim y_dim x_dim//2+1"]: ...


def rfftn(
    ift: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Complex[Array, "y_dim x_dim//2+1"] | Complex[Array, "z_dim y_dim x_dim//2+1"]:
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

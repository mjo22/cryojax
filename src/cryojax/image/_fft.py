"""
Routines to compute FFTs, in cryojax conventions.
"""

from typing import Any, Union, Optional

import jax.numpy as jnp

from ..typing import (
    RealImage,
    RealVolume,
    ComplexImage,
    ComplexVolume,
    Image,
    Volume,
)


def ifftn(
    ft: Union[Image, Volume],
    real: bool = False,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Union[Image, Volume]:
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


def fftn(
    ift: Union[Image, Volume],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Union[ComplexImage, ComplexVolume]:
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


def irfftn(
    ft: Union[ComplexImage, ComplexVolume],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Union[RealImage, RealVolume]:
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


def rfftn(
    ift: Union[RealImage, RealVolume],
    axes: Optional[tuple[int, ...]] = None,
    **kwargs: Any,
) -> Union[ComplexImage, ComplexVolume]:
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

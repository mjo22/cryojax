#!/usr/bin/env python
"""
Gaussian noise models
"""

__all__ = [
    "white_noise",
    "white_covariance",
    "lorenzian_noise",
    "lorenzian_covariance",
]


import numpy as np
from .fft import k_grid


def white_noise(shape: tuple, generator, sigma: float = 1.0) -> np.ndarray:
    """
    Generate gaussian random field from white noise.
    """
    size = np.prod(shape)
    noise = generator.normal(sigma=sigma, size=size).reshape(shape)

    return noise


def white_covariance(
    x: np.ndarray, y: np.ndarray, sigma: float = 1.0
) -> float:
    """
    Covariance function for white noise.
    """
    return 1 / sigma**2 if np.array_equal(x, y) else 0.0


def lorenzian_noise(
    shape: tuple,
    pixel_size: float,
    generator,
    sigma: float = 1.0,
    xi: float = None,
) -> np.ndarray:
    """
    Generate a gaussian random field with a given lorenzian,
    parameterized by a "temperature" and a correlation length.
    """
    # Check arguments
    assert len(shape) == 2
    Nx, Ny = shape
    xi = pixel_size if xi is None else xi
    # Generate coordinates
    kx, ky = k_grid(shape, pixel_size)
    kr = np.sqrt(kx**2 + ky**2)
    # Generate power spectrum
    spectrum = _lorenzian(kr, sigma, xi)
    spectrum[Nx // 2, Ny // 2] = 0
    # Generate noise in fourier space
    fourier_noise = generator.standard_normal(size=Nx * Ny).reshape(
        Nx, Ny
    ) + 1.0j * generator.standard_normal(size=Nx * Ny).reshape(Nx, Ny)
    fourier_noise *= spectrum
    # Go to real space, making sure we enforce f(k) = f(-k)
    noise = np.fft.irfft2(
        np.fft.fftshift(fourier_noise)[:, : Ny // 2 + Ny % 2], s=shape
    ).real

    return noise


def lorenzian_covariance(
    x: np.ndarray, y: np.ndarray, sigma: float = 1.0, xi: float = 1.0
) -> float:
    """
    Covariance function for a gaussian random field with
    a lorenzian power spectrum.
    """
    pass


def _lorenzian(k, sigma, xi):
    """
    Lorenzian function for a given wavenumber k.
    """
    return sigma**2 / (k**2 + np.divide(1, xi**2))


def k_grid(shape, pixel_size):
    """
    Create a k coordinate system with zero frequency
    component in the beginning.
    """
    ndim = len(shape)
    kcoords1D = []
    for i in range(ndim):
        ni = shape[i]
        ki = np.fft.fftshift(np.fft.fftfreq(ni)) * ni / pixel_size
        kcoords1D.append(ki)

    kcoords = np.meshgrid(*kcoords1D, indexing="ij")

    return kcoords


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    shape = (300, 300)
    pixel_size = 1.0
    generator = np.random.default_rng()
    noise = lorenzian_noise(shape, pixel_size, generator, sigma=1.0, xi=0.1)

    fig, ax = plt.subplots()
    ax.imshow(noise)
    fig.savefig("noise.png")

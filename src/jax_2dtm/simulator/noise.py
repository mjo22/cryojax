"""
Noise models for cryo-EM images.
"""

__all__ = [
    "Noise",
    "NullNoise",
    "GaussianNoise",
    "WhiteNoise",
    "EmpiricalNoise",
    "LorenzianNoise",
]

from abc import ABCMeta, abstractmethod

from .scattering import ImageConfig
from ..types import field, dataclass, Array


class Noise(metaclass=ABCMeta):
    """
    Base PyTree container for a noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.sample``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    # generator: = field(pytree_noise=False)

    @abstractmethod
    def sample(config: ImageConfig, freqs: Array) -> Array:
        """
        Sample a realization of the noise.
        """
        raise NotImplementedError


class GaussianNoise(Noise, metaclass=ABCMeta):
    """
    Base PyTree container for a gaussian noise model.

    When writing subclasses,

        1) Overwrite ``OpticsModel.variance``.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    def sample(config: ImageConfig, freqs: Array) -> Array:
        return 0.0

    @abstractmethod
    def variance(freqs: Array) -> Array:
        """
        The variance tensor of the gaussian. Only diagonal
        variances are supported.
        """
        raise NotImplementedError


@dataclass
class NullNoise(Noise):
    """
    This class can be used as a null noise model.
    """

    def sample(config: ImageConfig, freqs: Array) -> Array:
        return 0.0


@dataclass
class WhiteNoise(GaussianNoise):
    """
    Gaussian white noise (flat power spectrum).
    """

    def variance(freqs: Array) -> Array:
        raise NotImplementedError


@dataclass
class EmpiricalNoise(GaussianNoise):
    """
    Gaussian noise with an empirical power spectrum.
    """

    def variance(freqs: Array) -> Array:
        raise NotImplementedError


@dataclass
class LorenzianNoise(GaussianNoise):
    """
    Gaussian noise with a lorenzian power spectrum.
    """

    def variance(freqs: Array) -> Array:
        raise NotImplementedError


"""
__all__ = [
    "white_noise",
    "white_covariance",
    "lorenzian_noise",
    "lorenzian_covariance",
]


def white_noise(shape: tuple, generator, sigma: float = 1.0) -> np.ndarray:
    size = np.prod(shape)
    noise = generator.normal(sigma=sigma, size=size).reshape(shape)

    return noise


def white_covariance(
    x: np.ndarray, y: np.ndarray, sigma: float = 1.0
) -> float:
    return 1 / sigma**2 if np.array_equal(x, y) else 0.0


def lorenzian_noise(
    shape: tuple,
    pixel_size: float,
    generator,
    sigma: float = 1.0,
    xi: float = None,
) -> np.ndarray:
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
    pass


def _lorenzian(k, sigma, xi):
    return sigma**2 / (k**2 + np.divide(1, xi**2))


def k_grid(shape, pixel_size):
    ndim = len(shape)
    kcoords1D = []
    for i in range(ndim):
        ni = shape[i]
        ki = np.fft.fftshift(np.fft.fftfreq(ni)) * ni / pixel_size
        kcoords1D.append(ki)

    kcoords = np.meshgrid(*kcoords1D, indexing="ij")

    return kcoords
"""

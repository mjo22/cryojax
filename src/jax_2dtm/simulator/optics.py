"""
Models of instrument optics.
"""

from __future__ import annotations

__all__ = [
    "Optics",
    "NullOptics",
    "CTFOptics",
    "compute_ctf_power",
]

from abc import ABCMeta, abstractmethod

import jax.numpy as jnp

from ..types import dataclass, Array, Scalar
from ..core import Serializable


class Optics(Serializable, metaclass=ABCMeta):
    """
    Base PyTree container for an optics model. This
    is designed to compute an optics model in Fourier
    Space given some frequencies and parameters, and
    also store model parameters as a PyTree node.

    When writing subclasses,

        1) Overwrite the ``OpticsModel.compute`` method.
        2) Use the ``jax_2dtm.types.dataclass`` decorator.
    """

    @abstractmethod
    def compute(self, freqs: Array) -> Array:
        raise NotImplementedError

    def __call__(self, freqs: Array) -> Array:
        """
        Compute the optics model.

        Parameters
        ----------
        freqs : Array, shape `(N1, N2, 2)`
            The fourier wavevectors in the imaging plane.
        """
        return self.compute(freqs)


@dataclass
class NullOptics(Optics):
    """
    This class can be used as a null optics model.
    """

    def compute(self, freqs: Array) -> Array:
        return jnp.array(1.0)


@dataclass
class CTFOptics(Optics):
    """
    Compute a Contrast Transfer Function (CTF).

    Also acts as a PyTree container for the CTF parameters.
    See ``jax_2dtm.simulator.compute_ctf_power`` for more
    information.

    Attributes
    ----------
    defocus_u : `jax_2dtm.types.Scalar`
    defocus_v : `jax_2dtm.types.Scalar`
    defocus_angle : `jax_2dtm.types.Scalar`
    voltage : `jax_2dtm.types.Scalar`
    spherical_aberration : `jax_2dtm.types.Scalar`
    amplitude_contrast_ratio : `jax_2dtm.types.Scalar`
    phase_shift : `jax_2dtm.types.Scalar`
    b_factor : `jax_2dtm.types.Scalar`
    """

    defocus_u: Scalar = 10000.0
    defocus_v: Scalar = 10000.0
    defocus_angle: Scalar = 0.0
    voltage: Scalar = 300.0
    spherical_aberration: Scalar = 2.7
    amplitude_contrast_ratio: Scalar = 0.1
    phase_shift: Scalar = 0.0
    b_factor: Scalar = 1.0

    def compute(self, freqs: Array) -> Array:
        return compute_ctf_power(freqs, *self.iter_data())


def compute_ctf_power(
    freqs: Array,
    defocus_u: Scalar,
    defocus_v: Scalar,
    defocus_angle: Scalar,
    voltage: Scalar,
    spherical_aberration: Scalar,
    amplitude_contrast_ratio: Scalar,
    phase_shift: Scalar,
    b_factor: Scalar,
    *,
    normalize: bool = True,
) -> Array:
    """
    Computes CTF with given parameters.

    Parameters
    ----------
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The wave vectors in the imaging plane, in units
        of A.
    defocus_u : float
        The defocus in the major axis in Angstroms.
    defocus_v : float
        The defocus in the minor axis in Angstroms.
    defocus_angle : float
        The defocus angle in degree.
    voltage : float
        The accelerating voltage in kV.
    spherical_aberration : float
        The spherical aberration in mm.
    amplitude_contrast_ratio : float
        The amplitude contrast ratio.
    phase_shift : float
        The phase shift in degrees.
    b_factor : float, optional
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
    normalize : bool, optional
        Whether to normalize the CTF so that it has norm 1 in real space. Default is True.

    Returns
    -------
    ctf : `jax.Array`, shape `(N1, N2)`
        The contrast transfer function.
    """

    N1, N2 = freqs.shape[0:-1]
    theta = jnp.arctan2(freqs[..., 1], freqs[..., 0])
    k_sqr = jnp.sum(jnp.square(freqs), axis=-1)

    defocus = 0.5 * (
        defocus_u
        + defocus_v
        + (defocus_u - defocus_v) * jnp.cos(2 * (theta - defocus_angle))
    )

    lam = (12.2643 / (voltage + 0.97845e-6 * voltage * voltage)) ** 0.5
    gamma_defocus = -0.5 * defocus * lam * k_sqr
    gamma_sph = 0.25 * spherical_aberration * (lam**3) * (k_sqr**2)

    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift
    ctf = (1 - amplitude_contrast_ratio**2) ** 0.5 * jnp.sin(
        gamma
    ) - amplitude_contrast_ratio * jnp.cos(gamma)

    # We apply normalization before b-factor envelope
    if normalize:
        ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))

    ctf = ctf * jnp.exp(-0.25 * b_factor * k_sqr)

    return ctf

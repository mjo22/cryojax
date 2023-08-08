"""
Models of instrument optics.
"""

from __future__ import annotations

__all__ = [
    "Optics",
    "NullOptics",
    "CTFOptics",
    "compute_ctf",
]

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import jax.numpy as jnp

from ..core import dataclass, Array, Scalar, Serializable


class Optics(Serializable, metaclass=ABCMeta):
    """
    Base PyTree container for an optics model. This
    is designed to compute an optics model in Fourier
    Space given some frequencies and parameters, and
    also store model parameters as a PyTree node.

    When writing subclasses,

        1) Overwrite the ``OpticsModel.compute`` method.
        2) Use the ``cryojax.core.dataclass`` decorator.
    """

    @abstractmethod
    def compute(self, freqs: Array, **kwargs: Any) -> Array:
        raise NotImplementedError

    def __call__(self, freqs: Array, **kwargs: Any) -> Array:
        """
        Compute the optics model.

        Parameters
        ----------
        freqs : Array, shape `(N1, N2, 2)`
            The fourier wavevectors in the imaging plane.
        """
        return self.compute(freqs, **kwargs)


@dataclass
class NullOptics(Optics):
    """
    This class can be used as a null optics model.
    """

    def compute(self, freqs: Array, **kwargs: Any) -> Array:
        return jnp.array(1.0)


@dataclass
class CTFOptics(Optics):
    """
    Compute a Contrast Transfer Function (CTF).

    Also acts as a PyTree container for the CTF parameters.
    See ``cryojax.simulator.compute_ctf`` for more
    information.

    Attributes
    ----------
    defocus_u : `cryojax.core.Scalar`
    defocus_v : `cryojax.core.Scalar`
    defocus_angle : `cryojax.core.Scalar`
    voltage : `cryojax.core.Scalar`
    spherical_aberration : `cryojax.core.Scalar`
    amplitude_contrast_ratio : `cryojax.core.Scalar`
    phase_shift : `cryojax.core.Scalar`
    b_factor : `cryojax.core.Scalar`
    """

    defocus_u: Scalar = 10000.0
    defocus_v: Scalar = 10000.0
    defocus_angle: Scalar = 0.0
    voltage: Scalar = 300.0
    spherical_aberration: Scalar = 2.7
    amplitude_contrast: Scalar = 0.1
    phase_shift: Scalar = 0.0
    b_factor: Scalar = 1.0

    def compute(self, freqs: Array, **kwargs: Any) -> Array:
        return compute_ctf(freqs, *self.iter_data(), **kwargs)


def compute_ctf(
    freqs: Array,
    defocus_u: Scalar,
    defocus_v: Scalar,
    defocus_angle: Scalar,
    voltage: Scalar,
    spherical_aberration: Scalar,
    amplitude_contrast: Scalar,
    phase_shift: Scalar,
    b_factor: Optional[Scalar] = None,
    *,
    normalize: bool = True,
) -> Array:
    """
    Computes CTF with given parameters.

    Parameters
    ----------
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The wave vectors in the imaging plane, in units
        of 1/A.
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
    amplitude_contrast : float
        The amplitude contrast ratio.
    phase_shift : float
        The additional phase shift in radians.
    b_factor : float, optional
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
    normalize : bool, optional
        Whether to normalize the CTF so that it has norm 1 in real space.
        Default is True.

    Returns
    -------
    ctf : `jax.Array`, shape `(N1, N2)`
        The contrast transfer function.
    """

    # Unit conversions
    voltage *= 1000  # kV to V
    spherical_aberration *= 1e7  # mm to Angstroms

    N1, N2 = freqs.shape[0:-1]
    theta = jnp.arctan2(freqs[..., 1], freqs[..., 0])
    k_sqr = jnp.sum(jnp.square(freqs), axis=-1)

    defocus = 0.5 * (
        defocus_u
        + defocus_v
        + (defocus_u - defocus_v) * jnp.cos(2.0 * (theta - defocus_angle))
    )
    if jnp.isclose(jnp.abs(amplitude_contrast - 1.0), 1.0):
        ac = jnp.pi / 2
    else:
        ac = jnp.arctan(
            amplitude_contrast / jnp.sqrt(1.0 - amplitude_contrast**2)
        )

    lam = 12.2643 / (voltage + 0.97845e-6 * voltage**2) ** 0.5
    gamma_defocus = -0.5 * defocus * lam * k_sqr
    gamma_sph = 0.25 * spherical_aberration * (lam**3) * (k_sqr**2)
    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift - ac
    ctf = jnp.sin(gamma)

    # We apply normalization before b-factor envelope
    if normalize:
        ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))

    if b_factor is not None:
        ctf = ctf * jnp.exp(-0.25 * b_factor * k_sqr)

    return ctf

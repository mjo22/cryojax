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
from functools import partial

import jax
import jax.numpy as jnp

from ..utils import cartesian_to_polar
from ..core import dataclass, Array, ArrayLike, Parameter, CryojaxObject


@dataclass
class Optics(CryojaxObject, metaclass=ABCMeta):
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
    def compute(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        """Compute the optics model."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, ctf: ArrayLike, image: ArrayLike, **kwargs: Any) -> Array:
        """Apply the optics model."""
        raise NotImplementedError

    def __call__(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        """Compute the optics model."""
        return self.compute(freqs, **kwargs)


@dataclass
class NullOptics(Optics):
    """
    This class can be used as a null optics model.
    """

    def compute(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        return jnp.array(1.0)

    def apply(self, ctf: Array, image: ArrayLike, **kwargs: Any):
        return image


@dataclass
class CTFOptics(Optics):
    """
    Compute a Contrast Transfer Function (CTF).

    Also acts as a PyTree container for the CTF parameters.
    See ``cryojax.simulator.compute_ctf`` for more
    information.

    Attributes
    ----------
    defocus_u : `cryojax.core.Parameter`
    defocus_v : `cryojax.core.Parameter`
    defocus_angle : `cryojax.core.Parameter`
    voltage : `cryojax.core.Parameter`
    spherical_aberration : `cryojax.core.Parameter`
    amplitude_contrast_ratio : `cryojax.core.Parameter`
    phase_shift : `cryojax.core.Parameter`
    b_factor : `cryojax.core.Parameter`
    """

    defocus_u: Parameter = 10000.0
    defocus_v: Parameter = 10000.0
    defocus_angle: Parameter = 0.0
    voltage: Parameter = 300.0
    spherical_aberration: Parameter = 2.7
    amplitude_contrast: Parameter = 0.1
    phase_shift: Parameter = 0.0
    b_factor: Parameter = 1.0

    def apply(self, ctf: ArrayLike, image: ArrayLike, **kwargs: Any) -> Array:
        return ctf * image

    def compute(self, freqs: ArrayLike, **kwargs: Any) -> Array:
        return compute_ctf(freqs, *self.iter_data(), **kwargs)


@partial(jax.jit, static_argnames=["b_factor", "normalize", "degrees"])
def compute_ctf(
    freqs: ArrayLike,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    phase_shift: float,
    b_factor: Optional[float] = None,
    *,
    normalize: bool = True,
    degrees: bool = True,
) -> Array:
    """
    Computes CTF with given parameters.

    Parameters
    ----------
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The wave vectors in the imaging plane, in units
        of 1/A.
    defocus_u : `float`
        The defocus in the major axis in Angstroms.
    defocus_v : `float`
        The defocus in the minor axis in Angstroms.
    defocus_angle : `float`
        The defocus angle.
    voltage : `float`
        The accelerating voltage in kV.
    spherical_aberration : `float`
        The spherical aberration in mm.
    amplitude_contrast : `float`
        The amplitude contrast ratio.
    phase_shift : `float`
        The additional phase shift.
    b_factor : `float`, optional
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
    normalize : `bool`, optional
        Whether to normalize the CTF so that it has norm 1 in real space.
        Default is ``True``. The normalization is only applied when ``b_factor``
        is provided.
    degrees : `bool`, optional
        Whether or not the ``defocus_angle`` and ``phase_shift`` are given
        in degrees or radians.

    Returns
    -------
    ctf : `jax.Array`, shape `(N1, N2)`
        The contrast transfer function.
    """
    freqs = jnp.asarray(freqs)
    # Unit conversions
    voltage *= 1000  # kV to V
    spherical_aberration *= 1e7  # mm to Angstroms
    if degrees:  # degrees to radians
        phase_shift = jnp.deg2rad(phase_shift)
        defocus_angle = jnp.deg2rad(defocus_angle)

    # Polar coordinate system
    N1, N2 = freqs.shape[0:-1]
    k_sqr, theta = cartesian_to_polar(freqs, square=True)

    defocus = 0.5 * (
        defocus_u
        + defocus_v
        + (defocus_u - defocus_v) * jnp.cos(2.0 * (theta - defocus_angle))
    )
    ac = jnp.arctan(
        amplitude_contrast / jnp.sqrt(1.0 - amplitude_contrast**2)
    )

    lam = 12.2643 / (voltage + 0.97845e-6 * voltage**2) ** 0.5
    gamma_defocus = -0.5 * defocus * lam * k_sqr
    gamma_sph = 0.25 * spherical_aberration * (lam**3) * (k_sqr**2)
    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift - ac
    ctf = jnp.sin(gamma)

    # Apply b-factor envelope
    if b_factor is not None:
        # We apply normalization before b-factor envelope
        if normalize:
            ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))
        ctf = ctf * jnp.exp(-0.25 * b_factor * k_sqr)

    return ctf

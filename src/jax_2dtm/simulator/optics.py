"""
Routines for and definitions of models for instrument optics.
"""

from __future__ import annotations

__all__ = [
    "OpticsModel",
    "NullOptics",
    "CTFOptics",
    "OpticsImage",
    "compute_ctf_power",
]

import dataclasses
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

from .scattering import ScatteringImage
from ..types import dataclass, Array, Scalar

if TYPE_CHECKING:
    from .state import ParameterState


class OpticsModel(metaclass=ABCMeta):
    """
    Base class for an optics model. This
    is designed to compute an optics model in Fourier
    Space given some frequencies and parameters, and
    also store model parameters as a PyTree node.

    When writing subclasses,

        1) Overwrite the ``OpticsModel.compute`` method
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
class NullOptics(OpticsModel):
    """
    Base PyTree container for an optics model. This class can be
    used as a null model.
    """

    def compute(self, freqs: Array) -> Array:
        return jnp.array(1.0)


@dataclass
class CTFOptics(OpticsModel):
    """
    Compute a Contrast Transfer Function (CTF).

    Also acts as a PyTree container for the CTF parameters.

    Attributes
    ----------
    defocus_u : Scalar, `float` or shape `(M,)`
        The defocus in the major axis in Angstroms.
    defocus_v : Scalar, `float` or shape `(M,)`
        The defocus in the minor axis in Angstroms.
    defocus_angle : Scalar, `float` or shape `(M,)`
        The defocus angle in degree.
    voltage : Scalar, `float` or shape `(M,)`
        The spherical aberration in mm.
    spherical_aberration : Scalar, `float` or shape `(M,)`
        The accelerating voltage in kV.
    amplitude_contrast_ratio : Scalar, `float` or shape `(M,)`
        The amplitude contrast ratio.
    phase_shift : Scalar, `float` or shape `(M,)`
        The phase shift in degrees.
    b_factor : Scalar, `float` or shape `(M,)`
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
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


@dataclasses.dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the scattering pattern on the imaging plane,
    moduated by a CTF.
    """

    def render(self, state: "ParameterState") -> Array:
        """
        Render an image from a model of the CTF.
        """
        # Compute scattering at image plane.
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Compute and apply CTF
        ctf = state.optics(self.freqs)
        optics_image = ctf * scattering_image
        # Apply filters
        for filter in self.filters:
            optics_image = filter(optics_image)

        return optics_image


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
    normalize: bool = True,
) -> Array:
    """
    Computes CTF with given parameters.

    Parameters
    ----------
    freqs : Array, shape `(N1, N2, 2)`
        The wave vectors in the imaging plane.
    defocus_u : Scalar
        The defocus in the major axis in Angstroms.
    defocus_v : Scalar
        The defocus in the minor axis in Angstroms.
    defocus_angle : Scalar
        The defocus angle in degree.
    voltage : Scalar
        The accelerating voltage in kV.
    spherical_aberration : Scalar
        The spherical aberration in mm.
    amplitude_contrast_ratio : Scalar
        The amplitude contrast ratio.
    phase_shift : Scalar
        The phase shift in degrees.
    b_factor : Scalar, optional
        The B factor in A^2. If not provided, the B factor is assumed to be 0.
    normalize : bool, optional
        Whether to normalize the CTF so that it has norm 1 in real space. Default is True.

    Returns
    -------
    ctf : Array, shape `(N1, N2)`
        The contrast transfer function.
    """

    N1, N2 = freqs.shape[0:-1]
    theta = jnp.arctan2(freqs[..., 1], freqs[..., 0])
    kr_sqr = jnp.sum(jnp.square(freqs), axis=-1)

    defocus = 0.5 * (
        defocus_u
        + defocus_v
        + (defocus_u - defocus_v) * jnp.cos(2 * (theta - defocus_angle))
    )

    lam = (12.2643 / (voltage + 0.97845e-6 * voltage * voltage)) ** 0.5
    gamma_defocus = -0.5 * defocus * lam * kr_sqr
    gamma_sph = 0.25 * spherical_aberration * (lam**3) * (kr_sqr**2)

    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift
    ctf = (1 - amplitude_contrast_ratio**2) ** 0.5 * jnp.sin(
        gamma
    ) - amplitude_contrast_ratio * jnp.cos(gamma)

    # We apply normalization before b-factor envelope
    if normalize:
        ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))

    ctf = ctf * jnp.exp(-0.25 * b_factor * kr_sqr)

    return ctf

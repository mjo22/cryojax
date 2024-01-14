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

from abc import abstractmethod
from typing import Any, Optional, overload
from functools import partial

import jax
import jax.numpy as jnp
from equinox import Module

from .pose import Pose
from ..image import ParameterizedFilter
from ..core import field
from ..image import cartesian_to_polar
from ..typing import Real_, RealImage, Image, ImageCoords


class Optics(Module):
    """
    Base class for an optics model. This
    is designed to compute an optics model in Fourier
    space given some frequencies, and apply the model
    to an image.

    When writing subclasses,

        1) Overwrite the ``Optics.evaluate`` method.

    Attributes
    ----------
    envelope :
        A kernel that computes the envelope function of
        the optics model. By default, ``Gaussian()``.
    """

    envelope: Optional[ParameterizedFilter] = field(default=None)

    @overload
    @abstractmethod
    def evaluate(self, freqs: ImageCoords, pose: None, **kwargs: Any) -> Image:
        ...

    @overload
    @abstractmethod
    def evaluate(self, freqs: ImageCoords, pose: Pose, **kwargs: Any) -> Image:
        ...

    @abstractmethod
    def evaluate(
        self, freqs: ImageCoords, pose: Optional[Pose] = None, **kwargs: Any
    ) -> Image:
        """Compute the optics model."""
        raise NotImplementedError

    def __call__(
        self, freqs: ImageCoords, normalize: bool = True, **kwargs: Any
    ) -> Image:
        """Compute the optics model with an envelope."""
        if self.envelope is None:
            return self.evaluate(freqs, **kwargs)
        else:
            return self.envelope(freqs) * self.evaluate(
                freqs, normalize=normalize, **kwargs
            )


class NullOptics(Optics):
    """
    A null optics model.
    """

    envelope: Optional[ParameterizedFilter] = field(default=None)

    def evaluate(
        self, freqs: ImageCoords, pose: Optional[Pose] = None, **kwargs: Any
    ) -> RealImage:
        return jnp.ones(freqs.shape[0:-1], dtype=float)


class CTFOptics(Optics):
    """
    Compute a Contrast Transfer Function (CTF).

    See ``cryojax.simulator.compute_ctf`` for more
    information.

    Attributes
    ----------
    degrees :
    defocus_u :
    defocus_v :
    defocus_angle :
    voltage :
    spherical_aberration :
    amplitude_contrast_ratio :
    phase_shift :
    """

    degrees: bool = field(static=True, default=True)

    defocus_u: Real_ = field(default=10000.0)
    defocus_v: Real_ = field(default=10000.0)
    defocus_angle: Real_ = field(default=0.0)
    voltage: Real_ = field(default=300.0)
    spherical_aberration: Real_ = field(default=2.7)
    amplitude_contrast: Real_ = field(default=0.1)
    phase_shift: Real_ = field(default=0.0)

    def evaluate(
        self, freqs: ImageCoords, pose: Optional[Pose] = None, **kwargs: Any
    ) -> RealImage:
        defocus_offset = 0.0 if pose is None else pose.offset_z
        return compute_ctf(
            freqs,
            self.defocus_u + defocus_offset,
            self.defocus_v + defocus_offset,
            self.defocus_angle,
            self.voltage,
            self.spherical_aberration,
            self.amplitude_contrast,
            self.phase_shift,
            degrees=self.degrees,
            **kwargs,
        )


@partial(jax.jit, static_argnames=["normalize", "degrees"])
def compute_ctf(
    freqs: ImageCoords,
    defocus_u: Real_,
    defocus_v: Real_,
    defocus_angle: Real_,
    voltage: Real_,
    spherical_aberration: Real_,
    amplitude_contrast: Real_,
    phase_shift: Real_,
    *,
    normalize: bool = False,
    degrees: bool = True,
) -> RealImage:
    """
    Computes a real-valued CTF.

    Arguments
    ---------
    freqs :
        The wave vectors in the imaging plane, in units
        of 1/A.
    defocus_u :
        The defocus in the major axis in Angstroms.
    defocus_v :
        The defocus in the minor axis in Angstroms.
    defocus_angle :
        The defocus angle.
    voltage :
        The accelerating voltage in kV.
    spherical_aberration :
        The spherical aberration in mm.
    amplitude_contrast :
        The amplitude contrast ratio.
    phase_shift :
        The additional phase shift.
    normalize :
        Whether to normalize the CTF so that it has norm 1 in real space.
        Default is ``False``.
    degrees :
        Whether or not the ``defocus_angle`` and ``phase_shift`` are given
        in degrees or radians.

    Returns
    -------
    ctf :
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

    if normalize:
        N1, N2 = freqs.shape[0:-1]
        ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))

    return ctf

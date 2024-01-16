"""
Models of instrument optics.
"""

__all__ = ["CTF", "Optics", "NullOptics", "CTFOptics", "compute_ctf"]

from abc import abstractmethod
from typing import Optional
from functools import partial
from equinox import AbstractVar, Module

import jax
import jax.numpy as jnp

from .manager import ImageManager
from ..image import FourierOperatorLike, FourierOperator, Constant
from ..core import field
from ..image import cartesian_to_polar
from ..typing import Real_, RealImage, Image, ComplexImage, ImageCoords


class CTF(FourierOperator):
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

    defocus_u: Real_ = field(default=10000.0)
    defocus_v: Real_ = field(default=10000.0)
    defocus_angle: Real_ = field(default=0.0)
    voltage: Real_ = field(default=300.0)
    spherical_aberration: Real_ = field(default=2.7)
    amplitude_contrast: Real_ = field(default=0.1)
    phase_shift: Real_ = field(default=0.0)

    degrees: bool = field(static=True, default=True)

    def __call__(
        self, freqs: ImageCoords, defocus_offset: Real_ | float = 0.0
    ) -> RealImage:
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
        )


class Optics(Module):
    """
    Base class for an optics model.

    When writing subclasses,

        1) Overwrite the ``Optics.__call__`` method.
        2) Overwrite the ``Optics.ctf`` property.

    Attributes
    ----------
    ctf :
        The contrast transfer function model.
    envelope :
        A kernel that computes the envelope function of
        the optics model.
    normalize :
        Whether to normalize the CTF so that it has norm 1 in real space.
        Default is ``False``.
    """

    ctf: AbstractVar[FourierOperator]
    envelope: Optional[FourierOperatorLike] = field(default=None)

    normalize: bool = field(static=True, default=False)

    def evaluate(
        self, freqs: ImageCoords, defocus_offset: Real_ | float = 0.0
    ):
        """Evaluate the optics model. This is modeled as a contrast
        transfer function multiplied by an envelope function."""
        if self.envelope is None:
            ctf = self.ctf(freqs, defocus_offset=defocus_offset)
        else:
            ctf = self.envelope(freqs) * self.ctf(
                freqs, defocus_offset=defocus_offset
            )
        if self.normalize:
            N1, N2 = freqs.shape[0:-1]
            ctf = ctf / (jnp.linalg.norm(ctf) / jnp.sqrt(N1 * N2))

        return ctf

    @abstractmethod
    def __call__(
        self,
        image: ComplexImage,
        manager: ImageManager,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        """Pass an image through the optics model."""
        raise NotImplementedError


class NullOptics(Optics):
    """
    A null optics model.
    """

    ctf: FourierOperatorLike
    envelope: Optional[FourierOperatorLike] = field(default=None)

    def __init__(self):
        self.ctf = Constant(1.0)
        self.envelope = None
        self.normalize = False

    def __call__(
        self,
        image: ComplexImage,
        manager: ImageManager,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        return image


class CTFOptics(Optics):
    """
    An optics model with a real-valued contrast transfer function.
    """

    ctf: CTF = field(default_factory=CTF)
    envelope: Optional[FourierOperatorLike] = field(default=None)

    def __call__(
        self,
        image: ComplexImage,
        manager: ImageManager,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        """Compute the optics model with an envelope."""
        frequency_grid = manager.padded_frequency_grid_in_angstroms.get()
        return image * self.evaluate(
            frequency_grid, defocus_offset=defocus_offset
        )


@partial(jax.jit, static_argnames=["degrees"])
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

    return ctf

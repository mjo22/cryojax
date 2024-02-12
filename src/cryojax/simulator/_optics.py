"""
Models of instrument optics.
"""

from abc import abstractmethod
from typing import Optional
from functools import partial
from equinox import AbstractVar, Module, field

import jax
import jax.numpy as jnp

from ._config import ImageConfig
from ..image.operators import (
    FourierOperatorLike,
    AbstractFourierOperator,
    Constant,
)
from ..coordinates import cartesian_to_polar
from ..typing import Real_, RealImage, ComplexImage, Image, ImageCoords


class CTF(AbstractFourierOperator, strict=True):
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

    defocus_u: Real_ = field(default=10000.0, converter=jnp.asarray)
    defocus_v: Real_ = field(default=10000.0, converter=jnp.asarray)
    defocus_angle: Real_ = field(default=0.0, converter=jnp.asarray)
    voltage: Real_ = field(default=300.0, converter=jnp.asarray)
    spherical_aberration: Real_ = field(default=2.7, converter=jnp.asarray)
    amplitude_contrast: Real_ = field(default=0.1, converter=jnp.asarray)
    phase_shift: Real_ = field(default=0.0, converter=jnp.asarray)

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


class AbstractOptics(Module, strict=True):
    """
    Base class for an optics model.

    When writing subclasses,

        1) Overwrite the ``AbstractOptics.__call__`` method.
        2) Overwrite the ``AbstractOptics.ctf`` property.

    Attributes
    ----------
    ctf :
        The contrast transfer function model.
    envelope :
        The envelope function of the optics model.
    """

    ctf: AbstractVar[AbstractFourierOperator]
    envelope: AbstractVar[Optional[FourierOperatorLike]]

    @abstractmethod
    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        """Pass an image through the optics model."""
        raise NotImplementedError


class NullOptics(AbstractOptics):
    """
    A null optics model.
    """

    ctf: Constant
    envelope: None

    def __init__(self):
        self.ctf = Constant(1.0)
        self.envelope = None

    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        return fourier_potential_in_exit_plane


class WeakPhaseOptics(AbstractOptics, strict=True):
    """
    An optics model in the weak-phase approximation. Here, approximate the
    wavefunction by applying the CTF directly to the scattering potential.
    """

    ctf: CTF = field(default_factory=CTF)
    envelope: Optional[FourierOperatorLike] = field(default=None)

    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> ComplexImage:
        """Apply the CTF to the scattering potential."""
        N1, N2 = config.padded_shape
        frequency_grid = config.padded_frequency_grid_in_angstroms.get()
        # Compute the CTF
        if self.envelope is None:
            ctf = self.ctf(frequency_grid, defocus_offset=defocus_offset)
        else:
            ctf = self.envelope(frequency_grid) * self.ctf(
                frequency_grid, defocus_offset=defocus_offset
            )
        # ... compute the "contrast" as the CTF multiplied by the scattering potential
        fourier_contrast_in_detector_plane = ctf * fourier_potential_in_exit_plane

        return fourier_contrast_in_detector_plane


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
    ac = jnp.arctan(amplitude_contrast / jnp.sqrt(1.0 - amplitude_contrast**2))

    wavelength = 12.2643 / (voltage + 0.97845e-6 * voltage**2) ** 0.5
    gamma_defocus = -0.5 * defocus * wavelength * k_sqr
    gamma_sph = 0.25 * spherical_aberration * (wavelength**3) * (k_sqr**2)
    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift - ac
    ctf = jnp.sin(gamma)

    return ctf

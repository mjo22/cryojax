"""
Models of instrument optics.
"""

from abc import abstractmethod
from typing import ClassVar, Optional
from typing_extensions import override
from equinox import AbstractClassVar, Module, field

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
    """Compute the Contrast Transfer Function (CTF) in for a weakly
    scattering specimen.

    **Attributes:**

    `defocus_u`: The major axis defocus in Angstroms.

    `defocus_v`: The minor axis defocus in Angstroms.

    `astigmatism_angle`: The defocus angle.

    `voltage_in_kv`: The accelerating voltage in kV.

    `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.

    `amplitude_contrast_ratio`: The amplitude contrast ratio.

    `phase_shift`: The additional phase shift.

    `degrees`: Whether or not the `astigmatism_angle` and `phase_shift` are given
              in degrees or radians.
    """

    defocus_u: Real_ = field(default=10000.0, converter=jnp.asarray)
    defocus_v: Real_ = field(default=10000.0, converter=jnp.asarray)
    astigmatism_angle: Real_ = field(default=0.0, converter=jnp.asarray)
    voltage_in_kilovolts: Real_ = field(default=300.0, converter=jnp.asarray)
    spherical_aberration_in_mm: Real_ = field(default=2.7, converter=jnp.asarray)
    amplitude_contrast_ratio: Real_ = field(default=0.1, converter=jnp.asarray)
    phase_shift: Real_ = field(default=0.0, converter=jnp.asarray)

    degrees: bool = field(static=True, default=True)

    @property
    def wavelength_in_angstroms(self):
        voltage_in_volts = 1000.0 * self.voltage_in_kilovolts  # kV to V
        return 12.2643 / (voltage_in_volts + 0.97845e-6 * voltage_in_volts**2) ** 0.5

    def __call__(
        self, freqs: ImageCoords, defocus_offset: Real_ | float = 0.0
    ) -> RealImage:
        if self.degrees:  # degrees to radians
            phase_shift = jnp.deg2rad(self.phase_shift)
            astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        return _compute_ctf(
            freqs,
            self.defocus_u + jnp.asarray(defocus_offset),
            self.defocus_v + jnp.asarray(defocus_offset),
            astigmatism_angle,
            self.wavelength_in_angstroms,
            self.spherical_aberration_in_mm * 1e7,  # mm to Angstroms
            self.amplitude_contrast_ratio,
            phase_shift,
        )


class AbstractOptics(Module, strict=True):
    """Base class for an optics model.

    **Attributes:**

    `ctf`: The contrast transfer function model.

    `envelope`: The envelope function of the optics model.

    `is_linear`: If `True`, the optics model directly computes
                 the image contrast from the potential. If `False`,
                 the optics model computes the wavefunction.
    """

    ctf: CTF
    envelope: FourierOperatorLike

    is_linear: AbstractClassVar[bool]

    def __init__(
        self,
        ctf: CTF,
        envelope: Optional[FourierOperatorLike] = None,
    ):
        self.ctf = ctf
        self.envelope = envelope or Constant(1.0)

    @property
    def wavelength_in_angstroms(self) -> Real_:
        return self.ctf.wavelength_in_angstroms

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
    """A null optics model."""

    is_linear: ClassVar[bool] = True

    @override
    def __init__(self):
        self.ctf = CTF()
        self.envelope = Constant(1.0)

    @override
    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> Image:
        return fourier_potential_in_exit_plane


class WeakPhaseOptics(AbstractOptics, strict=True):
    """An optics model in the weak-phase approximation. Here, compute the image
    contrast by applying the CTF directly to the scattering potential.
    """

    is_linear: ClassVar[bool] = True

    @override
    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> ComplexImage:
        """Apply the CTF directly to the scattering potential."""
        N1, N2 = config.padded_shape
        frequency_grid = config.padded_frequency_grid_in_angstroms.get()
        # Compute the CTF
        ctf = self.envelope(frequency_grid) * self.ctf(
            frequency_grid, defocus_offset=defocus_offset
        )
        # ... compute the contrast as the CTF multiplied by the scattering potential
        fourier_contrast_in_detector_plane = ctf * fourier_potential_in_exit_plane

        return fourier_contrast_in_detector_plane


def _compute_ctf(
    freqs: ImageCoords,
    defocus_u: Real_,
    defocus_v: Real_,
    astigmatism_angle: Real_,
    wavelength: Real_,
    spherical_aberration: Real_,
    amplitude_contrast_ratio: Real_,
    phase_shift: Real_,
) -> RealImage:
    k_sqr, theta = cartesian_to_polar(freqs, square=True)
    defocus = 0.5 * (
        defocus_u
        + defocus_v
        + (defocus_u - defocus_v) * jnp.cos(2.0 * (theta - astigmatism_angle))
    )
    ac = jnp.arctan(
        amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
    )
    gamma_defocus = -0.5 * defocus * wavelength * k_sqr
    gamma_sph = 0.25 * spherical_aberration * (wavelength**3) * (k_sqr**2)
    gamma = (2 * jnp.pi) * (gamma_defocus + gamma_sph) - phase_shift - ac
    ctf = jnp.sin(gamma)

    return ctf

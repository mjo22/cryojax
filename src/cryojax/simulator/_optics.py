"""
Models of instrument optics.
"""

from abc import abstractmethod
from typing import ClassVar, Optional
from typing_extensions import override
from equinox import AbstractClassVar, AbstractVar, Module, field

import jax.numpy as jnp

from ._config import ImageConfig
from ..image.operators import (
    FourierOperatorLike,
    AbstractFourierOperatorInAngstroms,
    Constant,
)
from ..coordinates import cartesian_to_polar
from ..typing import Real_, RealImage, ComplexImage, Image, ImageCoords
from ..core import error_if_negative, error_if_not_positive, error_if_not_fractional


class CTF(AbstractFourierOperatorInAngstroms, strict=True):
    """Compute the Contrast Transfer Function (CTF) in for a weakly
    scattering specimen.

    **Attributes:**

    `defocus_u_in_angstroms`: The major axis defocus in Angstroms.

    `defocus_v_in_angstroms`: The minor axis defocus in Angstroms.

    `astigmatism_angle`: The defocus angle.

    `voltage_in_kv`: The accelerating voltage in kV.

    `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.

    `amplitude_contrast_ratio`: The amplitude contrast ratio.

    `phase_shift`: The additional phase shift.
    """

    defocus_u_in_angstroms: Real_ = field(
        default=10000.0, converter=error_if_not_positive
    )
    defocus_v_in_angstroms: Real_ = field(
        default=10000.0, converter=error_if_not_positive
    )
    astigmatism_angle: Real_ = field(default=0.0, converter=jnp.asarray)
    voltage_in_kilovolts: Real_ = field(default=300.0, converter=error_if_not_positive)
    spherical_aberration_in_mm: Real_ = field(default=2.7, converter=error_if_negative)
    amplitude_contrast_ratio: Real_ = field(
        default=0.1, converter=error_if_not_fractional
    )
    phase_shift: Real_ = field(default=0.0, converter=jnp.asarray)

    @property
    def wavelength_in_angstroms(self):
        voltage_in_volts = 1000.0 * self.voltage_in_kilovolts  # kV to V
        return 12.2643 / (voltage_in_volts + 0.97845e-6 * voltage_in_volts**2) ** 0.5

    def __call__(
        self,
        frequency_grid_in_angstroms: ImageCoords,
        defocus_offset: Real_ | float = 0.0,
    ) -> RealImage:
        # Convert degrees to radians
        phase_shift = jnp.deg2rad(self.phase_shift)
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Compute phase shifts for CTF
        phase_shifts = _compute_phase_shifts(
            frequency_grid_in_angstroms,
            self.defocus_u_in_angstroms + jnp.asarray(defocus_offset),
            self.defocus_v_in_angstroms + jnp.asarray(defocus_offset),
            astigmatism_angle,
            self.wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
            self.amplitude_contrast_ratio,
            phase_shift,
        )
        return jnp.sin(phase_shifts).at[0, 0].set(0.0)


class AbstractOptics(Module, strict=True):
    """Base class for an optics model.

    **Attributes:**

    `ctf`: The contrast transfer function model.

    `envelope`: The envelope function of the optics model.

    `is_linear`: If `True`, the optics model directly computes
                 the image contrast from the potential. If `False`,
                 the optics model computes the wavefunction.
    """

    ctf: AbstractVar[CTF]
    envelope: AbstractVar[FourierOperatorLike]

    is_linear: AbstractClassVar[bool]

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

    ctf: CTF
    envelope: FourierOperatorLike

    is_linear: ClassVar[bool] = True

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

    ctf: CTF
    envelope: FourierOperatorLike

    is_linear: ClassVar[bool] = True

    def __init__(
        self,
        ctf: CTF,
        envelope: Optional[FourierOperatorLike] = None,
    ):
        self.ctf = ctf
        self.envelope = envelope or Constant(1.0)

    @override
    def __call__(
        self,
        fourier_potential_in_exit_plane: ComplexImage,
        config: ImageConfig,
        defocus_offset: Real_ | float = 0.0,
    ) -> ComplexImage:
        """Apply the CTF directly to the scattering potential."""
        frequency_grid = config.wrapped_padded_frequency_grid_in_angstroms.get()
        # Compute the CTF
        ctf = self.envelope(frequency_grid) * self.ctf(
            frequency_grid, defocus_offset=defocus_offset
        )
        # ... compute the contrast as the CTF multiplied by the scattering potential
        fourier_contrast_in_detector_plane = ctf * fourier_potential_in_exit_plane

        return fourier_contrast_in_detector_plane


def _compute_phase_shifts(
    frequency_grid_in_angstroms: ImageCoords,
    defocus_u_in_angstroms: Real_,
    defocus_v_in_angstroms: Real_,
    astigmatism_angle: Real_,
    wavelength_in_angstroms: Real_,
    spherical_aberration_in_angstroms: Real_,
    amplitude_contrast_ratio: Real_,
    phase_shift: Real_,
) -> RealImage:
    k_sqr, azimuth = cartesian_to_polar(frequency_grid_in_angstroms, square=True)
    defocus = 0.5 * (
        defocus_u_in_angstroms
        + defocus_v_in_angstroms
        + (defocus_u_in_angstroms - defocus_v_in_angstroms)
        * jnp.cos(2.0 * (azimuth - astigmatism_angle))
    )
    amplitude_contrast_phase_shifts = jnp.arctan(
        amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
    )
    defocus_phase_shifts = -0.5 * defocus * wavelength_in_angstroms * k_sqr
    aberration_phase_shifts = (
        0.25
        * spherical_aberration_in_angstroms
        * (wavelength_in_angstroms**3)
        * (k_sqr**2)
    )
    phase_shifts = (
        (2 * jnp.pi) * (defocus_phase_shifts + aberration_phase_shifts)
        - phase_shift
        - amplitude_contrast_phase_shifts
    )

    return phase_shifts

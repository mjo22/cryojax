from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...image.operators import FourierOperatorLike
from ...internal import error_if_not_fractional
from .._instrument_config import InstrumentConfig
from .transfer_function import AbstractCTF


class AbstractTransferTheory(eqx.Module, strict=True):
    """A transfer theory for the weak-phase approximation. This class
    propagates the fourier spectrum of the object from a plane directly below it to
    the plane of the detector. In other terms, it computes a noiseless cryo-EM
    image from a 2D projection.
    """

    ctf: eqx.AbstractVar[AbstractCTF]


class ContrastTransferTheory(AbstractTransferTheory, strict=True):
    """A transfer theory for the weak-phase approximation. This class
    propagates the fourier spectrum of the object from a plane directly below it to
    the plane of the detector. In other terms, it computes a noiseless cryo-EM
    image from a 2D projection.
    """

    ctf: AbstractCTF
    envelope: Optional[FourierOperatorLike]
    amplitude_contrast_ratio: Float[Array, ""]
    phase_shift: Float[Array, ""]

    def __init__(
        self,
        ctf: AbstractCTF,
        envelope: Optional[FourierOperatorLike] = None,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
        phase_shift: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `ctf`: The contrast transfer function model.
        - `envelope`: The envelope function of the optics model.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        - `phase_shift`: The additional phase shift.
        """

        self.ctf = ctf
        self.envelope = envelope
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)
        self.phase_shift = jnp.asarray(phase_shift)

    def propagate_object_to_detector_plane(
        self,
        object_spectrum_at_exit_plane: (
            Complex[
                Array,
                "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
            ]
            | Complex[
                Array,
                "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
            ]
        ),
        instrument_config: InstrumentConfig,
        *,
        defocus_offset: Optional[Float[Array, ""] | float] = None,
        is_projection_approximation: bool = True,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Apply the CTF directly to the phase shifts in the exit plane.

        **Arguments:**

        - `object_spectrum_at_exit_plane`:
            The fourier spectrum of the scatterer phase shifts in a plane directly
            below it.
        - `instrument_config`:
            The configuration of the resulting image.
        - `is_projection_approximation`:
            If `True`, the `object_spectrum_in_exit_plane` is a projection
            approximation and is therefore the fourier transform of a real-valued
            array. If `False`, `object_spectrum_in_exit_plane` is extracted from
            the ewald sphere and is therefore the fourier transform of a complex-valued
            array.
        - `defocus_offset`:
            An optional defocus offset to apply to the CTF defocus at
            runtime.
        """
        frequency_grid = instrument_config.padded_frequency_grid_in_angstroms
        if is_projection_approximation:
            # Compute the CTF, including additional phase shifts
            ctf_array = self.ctf(
                frequency_grid,
                voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
                phase_shift=self.phase_shift,
                amplitude_contrast_ratio=self.amplitude_contrast_ratio,
                outputs_exp=False,
                defocus_offset=defocus_offset,
            )
            # ... compute the contrast as the CTF multiplied by the exit plane
            # phase shifts
            contrast_spectrum_at_detector_plane = (
                ctf_array * object_spectrum_at_exit_plane
            )
        else:
            # Propagate to the exit plane when the phase spectrum is
            # the surface of the ewald sphere
            aberration_phase_shifts = self.ctf.compute_aberration_phase_shifts(
                frequency_grid,
                voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
                defocus_offset=defocus_offset,
            ) - jnp.deg2rad(self.phase_shift)
            contrast_spectrum_at_detector_plane = _compute_contrast_from_ewald_sphere(
                object_spectrum_at_exit_plane,
                aberration_phase_shifts,
                self.amplitude_contrast_ratio,
                instrument_config,
            )
        if self.envelope is not None:
            contrast_spectrum_at_detector_plane *= self.envelope(frequency_grid)

        return contrast_spectrum_at_detector_plane


class WaveTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model that propagates the exit wave to the detector plane."""

    ctf: AbstractCTF

    def __init__(
        self,
        ctf: AbstractCTF,
    ):
        """**Arguments:**

        - `ctf`: The contrast transfer function model.
        """

        self.ctf = ctf

    def propagate_wavefunction_to_detector_plane(
        self,
        wavefunction_spectrum_at_exit_plane: Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ],
        instrument_config: InstrumentConfig,
        *,
        defocus_offset: Optional[Float[Array, ""] | float] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        """Apply the wave transfer function to the wavefunction in the exit plane."""
        frequency_grid = instrument_config.padded_full_frequency_grid_in_angstroms
        # Compute the wave transfer function
        ctf_array = self.ctf(
            frequency_grid,
            voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
            outputs_exp=True,
            defocus_offset=defocus_offset,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        wavefunction_spectrum_at_detector_plane = (
            ctf_array * wavefunction_spectrum_at_exit_plane
        )

        return wavefunction_spectrum_at_detector_plane


def _compute_contrast_from_ewald_sphere(
    object_spectrum_at_exit_plane,
    aberration_phase_shifts,
    amplitude_contrast_ratio,
    instrument_config,
):
    cos, sin = jnp.cos(aberration_phase_shifts), jnp.sin(aberration_phase_shifts)
    ac = amplitude_contrast_ratio
    # Compute the contrast, breaking the computation into positive and
    # negative frequencies
    y_dim, x_dim = instrument_config.padded_y_dim, instrument_config.padded_x_dim
    # ... first handle the grid of frequencies
    pos_object_yx = object_spectrum_at_exit_plane[1:, 1 : x_dim // 2 + x_dim % 2]
    neg_object_yx = jnp.flip(
        jnp.flip(object_spectrum_at_exit_plane[1:, x_dim // 2 + x_dim % 2 :], axis=-1),
        axis=0,
    )
    if x_dim % 2 == 0:
        pos_object_yx = jnp.concatenate(
            (pos_object_yx, neg_object_yx[:, -1, None].conj()), axis=-1
        )
    contrast_yx = _ewald_propagate_kernel(
        pos_object_yx,
        neg_object_yx,
        ac,
        sin[1:, 1:],
        cos[1:, 1:],
    )
    # ... next handle the line of frequencies at y = 0
    pos_object_0x = object_spectrum_at_exit_plane[0, 1 : x_dim // 2 + x_dim % 2]
    neg_object_0x = jnp.flip(
        object_spectrum_at_exit_plane[0, x_dim // 2 + x_dim % 2 :], axis=-1
    )
    if x_dim % 2 == 0:
        pos_object_0x = jnp.concatenate(
            (pos_object_0x, neg_object_0x[-1, None].conj()), axis=0
        )
    contrast_0x = _ewald_propagate_kernel(
        pos_object_0x,
        neg_object_0x,
        ac,
        sin[0, 1 : x_dim // 2 + 1],
        cos[0, 1 : x_dim // 2 + 1],
    )
    # ... then handle the line of frequencies at x = 0
    pos_object_y0 = object_spectrum_at_exit_plane[1 : y_dim // 2 + y_dim % 2, 0]
    neg_object_y0 = jnp.flip(
        object_spectrum_at_exit_plane[y_dim // 2 + y_dim % 2 :, 0], axis=-1
    )
    if y_dim % 2 == 0:
        pos_object_y0 = jnp.concatenate(
            (pos_object_y0, neg_object_y0[-1, None].conj()), axis=0
        )
    contrast_y0 = _ewald_propagate_kernel(
        pos_object_y0,
        neg_object_y0,
        ac,
        sin[1 : y_dim // 2 + 1, 0],
        cos[1 : y_dim // 2 + 1, 0],
    )
    # ... concatenate the zero mode to the line of frequencies at x = 0
    object_00 = object_spectrum_at_exit_plane[0, 0]
    contrast_00 = _ewald_propagate_kernel(
        object_00,
        object_00,
        ac,
        sin[0, 0],
        cos[0, 0],
    )
    contrast_y0 = jnp.concatenate(
        (
            contrast_00[None],
            (contrast_y0 if y_dim % 2 == 1 else contrast_y0[:-1]),
            jnp.flip(contrast_y0.conjugate()),
        ),
        axis=0,
    )
    # ... concatenate the results
    contrast_yx = jnp.concatenate((contrast_0x[None, :], contrast_yx), axis=0)
    contrast_spectrum_at_detector_plane = jnp.concatenate(
        (contrast_y0[:, None], contrast_yx), axis=1
    )

    return contrast_spectrum_at_detector_plane


def _ewald_propagate_kernel(pos, neg, ac, sin, cos):
    w1, w2 = ac, jnp.sqrt(1 - ac**2)
    return 0.5 * (
        (w2 * (pos.real + neg.real) + w1 * (pos.imag + neg.imag)) * sin
        + (w2 * (pos.imag + neg.imag) - w1 * (pos.real + neg.real)) * cos
        + 1.0j
        * (
            (w2 * (pos.imag - neg.imag) + w1 * (neg.real - pos.real)) * sin
            + (w2 * (neg.real - pos.real) + w1 * (neg.imag - pos.imag)) * cos
        )
    )

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
        object_phase_spectrum_at_exit_plane: (
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
        is_projection_approximation: bool = True,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Apply the CTF directly to the phase shifts in the exit plane.

        **Arguments:**

        - `object_phase_spectrum_at_exit_plane`:
            The fourier spectrum of the object phase shifts in a plane directly
            below it.
        - `instrument_config`:
            The configuration of the resulting image.
        - `is_projection_approximation`:
            If `True`, the `object_phase_spectrum_in_exit_plane` is a projection
            approximation and is therefore the fourier transform of a real-valued
            array. If `False`, `object_phase_spectrum_in_exit_plane` is extracted from
            the ewald sphere and is therefore the fourier transform of a complex-valued
            array.
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
            )
            # ... compute the contrast as the CTF multiplied by the exit plane
            # phase shifts
            contrast_spectrum_at_detector_plane = (
                ctf_array * object_phase_spectrum_at_exit_plane
            )
        else:
            # Propagate to the exit plane when the object spectrum is
            # the surface of the ewald sphere
            aberration_phase_shifts = self.ctf.compute_aberration_phase_shifts(
                frequency_grid,
                voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
            ) - jnp.deg2rad(self.phase_shift)
            contrast_spectrum_at_detector_plane = _compute_contrast_from_ewald_sphere(
                object_phase_spectrum_at_exit_plane,
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
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        wavefunction_spectrum_at_detector_plane = (
            ctf_array * wavefunction_spectrum_at_exit_plane
        )

        return wavefunction_spectrum_at_detector_plane


def _compute_contrast_from_ewald_sphere(
    object_phase_spectrum_at_exit_plane,
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
    pos_object_yx = object_phase_spectrum_at_exit_plane[:, 1 : x_dim // 2 + x_dim % 2]
    neg_object_yx = jnp.flip(
        object_phase_spectrum_at_exit_plane[:, x_dim // 2 + x_dim % 2 :], axis=-1
    )
    contrast_yx = _ewald_propagate_kernel(
        (pos_object_yx if x_dim % 2 == 1 else jnp.pad(pos_object_yx, ((0, 0), (0, 1)))),
        neg_object_yx,
        ac,
        sin[:, 1:],
        cos[:, 1:],
    )
    # ... next handle the line of frequencies at x = 0
    pos_object_y0 = object_phase_spectrum_at_exit_plane[1 : y_dim // 2 + y_dim % 2, 0]
    neg_object_y0 = jnp.flip(
        object_phase_spectrum_at_exit_plane[y_dim // 2 + y_dim % 2 :, 0], axis=-1
    )
    contrast_y0 = _ewald_propagate_kernel(
        (pos_object_y0 if y_dim % 2 == 1 else jnp.pad(pos_object_y0, ((0, 1),))),
        neg_object_y0,
        ac,
        sin[1 : y_dim // 2 + 1, 0],
        cos[1 : y_dim // 2 + 1, 0],
    )
    # ... concatenate the zero mode to the line of frequencies at x = 0
    object_00 = object_phase_spectrum_at_exit_plane[0, 0]
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
    contrast_spectrum_at_detector_plane = 0.5 * jnp.concatenate(
        (contrast_y0[:, None], contrast_yx), axis=1
    )

    return contrast_spectrum_at_detector_plane


def _ewald_propagate_kernel(neg, pos, ac, sin, cos):
    return (
        (neg.real + pos.real + ac * (neg.imag + pos.imag)) * sin
        + (neg.imag + pos.imag - ac * (neg.real + pos.real)) * cos
        + 1.0j
        * (
            (pos.imag - neg.imag + ac * (neg.real - pos.real)) * sin
            + (neg.real - pos.real + ac * (neg.imag - pos.imag)) * cos
        )
    )

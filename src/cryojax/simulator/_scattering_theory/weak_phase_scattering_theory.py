from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...utils import get_filter_spec
from .._instrument_config import InstrumentConfig
from .._pose import AbstractPose
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractIce
from .._structural_ensemble import (
    AbstractAssembly,
    AbstractConformationalVariable,
    AbstractStructuralEnsemble,
)
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractScatteringTheory
from .common_functions import convert_units_of_integrated_potential


class AbstractWeakPhaseScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a scattering theory in linear image formation theory
    (the weak-phase approximation).
    """

    @abstractmethod
    def compute_object_spectrum_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = instrument_config.padded_shape
        # ... compute the squared wavefunction directly from the image contrast
        # as |psi|^2 = 1 + 2C.
        contrast_spectrum_at_detector_plane = (
            self.compute_contrast_spectrum_at_detector_plane(instrument_config, rng_key)
        )
        intensity_spectrum_at_detector_plane = (
            (2 * contrast_spectrum_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
        )
        return intensity_spectrum_at_detector_plane


class WeakPhaseScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Base linear image formation theory."""

    structural_ensemble: AbstractStructuralEnsemble
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce] = None

    def __init__(
        self,
        structural_ensemble: AbstractStructuralEnsemble,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractIce] = None,
    ):
        """**Arguments:**

        - `structural_ensemble`: The structural ensemble of scattering potentials.
        - `potential_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.structural_ensemble = structural_ensemble
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    def __check_init__(self):
        if not self.potential_integrator.is_projection_approximation:
            cls = type(self.potential_integrator).__name__
            raise NotImplementedError(
                "`WeakPhaseScatteringTheory` does not currently support "
                f"`potential_integrator = {cls}(...)` as this is not a projection "
                "approximation, i.e. it returns a complex-valued array in real space. "
                "In order to use this integrator, try using the "
                "`HighEnergyScatteringTheory`."
            )

    @override
    def compute_object_spectrum_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # Compute the phase shifts in the exit plane
        object_spectrum_at_exit_plane = (
            _compute_object_spectrum_from_scattering_potential(
                self.structural_ensemble, self.potential_integrator, instrument_config
            )
        )
        # ... apply in-plane translation
        translational_phase_shifts = self.structural_ensemble.pose.compute_shifts(
            instrument_config.padded_frequency_grid_in_angstroms
        )
        object_spectrum_at_exit_plane *= translational_phase_shifts

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                object_spectrum_at_exit_plane = (
                    self.solvent.compute_object_spectrum_with_ice(
                        rng_key, object_spectrum_at_exit_plane, instrument_config
                    )
                )

        return object_spectrum_at_exit_plane

    @override
    def compute_contrast_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        object_spectrum_at_exit_plane = self.compute_object_spectrum_at_exit_plane(
            instrument_config, rng_key
        )
        contrast_spectrum_at_detector_plane = self.transfer_theory(
            object_spectrum_at_exit_plane,
            instrument_config,
        )

        return contrast_spectrum_at_detector_plane


class LinearSuperpositionScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Compute the superposition of images over a batch of poses and potentials
    parameterized by an `AbstractAssembly`. This must operate in the weak phase
    approximation.
    """

    assembly: AbstractAssembly
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce] = None

    def __init__(
        self,
        assembly: AbstractAssembly,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractIce] = None,
    ):
        """**Arguments:**

        - `assembly`: An concrete class of an `AbstractAssembly`. This is used to
              output a batch of states over which to
              compute a superposition of images.
        - `potential_integrator`: The method for integrating the specimen potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.assembly = assembly
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    @override
    def compute_object_spectrum_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        def compute_image(ensemble_mapped, ensemble_no_mapped, instrument_config):
            ensemble = eqx.combine(ensemble_mapped, ensemble_no_mapped)
            object_spectrum_at_exit_plane = (
                _compute_object_spectrum_from_scattering_potential(
                    ensemble, self.potential_integrator, instrument_config
                )
            )
            translational_phase_shifts = ensemble.pose.compute_shifts(
                instrument_config.padded_frequency_grid_in_angstroms
            )
            return translational_phase_shifts * object_spectrum_at_exit_plane

        @eqx.filter_jit
        def compute_image_superposition(
            ensemble_mapped, ensemble_no_mapped, instrument_config
        ):
            return jnp.sum(
                jax.lax.map(
                    lambda x: compute_image(x, ensemble_no_mapped, instrument_config),
                    ensemble_mapped,
                ),
                axis=0,
            )

        # Get the batch
        ensemble_batch, _ = self.assembly.get_subcomponents_and_z_positions_in_lab_frame()
        # Setup vmap over the pose and conformation
        is_mapped = lambda x: isinstance(
            x, (AbstractPose, AbstractConformationalVariable)
        )
        to_mapped = jax.tree_util.tree_map(is_mapped, ensemble_batch, is_leaf=is_mapped)
        mapped, no_mapped = eqx.partition(ensemble_batch, to_mapped)

        object_spectrum_at_exit_plane = compute_image_superposition(
            mapped, no_mapped, instrument_config
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                object_spectrum_at_exit_plane = (
                    self.solvent.compute_object_spectrum_with_ice(
                        rng_key, object_spectrum_at_exit_plane, instrument_config
                    )
                )

        return object_spectrum_at_exit_plane

    @override
    def compute_contrast_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        def compute_image(pytree_vmap, pytree_novmap, instrument_config):
            ensemble_vmap, transfer_vmap = pytree_vmap
            ensemble_novmap, transfer_novmap = pytree_novmap
            ensemble = eqx.combine(ensemble_vmap, ensemble_novmap)
            transfer_theory = eqx.combine(transfer_vmap, transfer_novmap)
            object_spectrum_at_exit_plane = (
                _compute_object_spectrum_from_scattering_potential(
                    ensemble, self.potential_integrator, instrument_config
                )
            )
            translational_phase_shifts = ensemble.pose.compute_shifts(
                instrument_config.padded_frequency_grid_in_angstroms
            )
            contrast_spectrum_at_detector_plane = transfer_theory(
                object_spectrum_at_exit_plane, instrument_config
            )

            return translational_phase_shifts * contrast_spectrum_at_detector_plane

        @eqx.filter_jit
        def compute_image_superposition(pytree_vmap, pytree_novmap, instrument_config):
            return jnp.sum(
                jax.lax.map(
                    lambda x: compute_image(x, pytree_novmap, instrument_config),
                    pytree_vmap,
                ),
                axis=0,
            )

        # Get the batches
        ensemble_batch, z_positions = (
            self.assembly.get_subcomponents_and_z_positions_in_lab_frame()
        )
        transfer_theory_batch = eqx.tree_at(
            lambda x: x.ctf.defocus_in_angstroms,
            self.transfer_theory,
            self.transfer_theory.ctf.defocus_in_angstroms + z_positions,
        )
        # Setup vmap over the pose and conformation
        is_vmapped = lambda x: isinstance(
            x, (AbstractPose, AbstractConformationalVariable)
        )
        filter_spec_for_ensemble = jax.tree_util.tree_map(
            is_vmapped, ensemble_batch, is_leaf=is_vmapped
        )
        ensemble_vmap, ensemble_novmap = eqx.partition(
            ensemble_batch, filter_spec_for_ensemble
        )
        # ... setup vmap over the CTF
        filter_spec_for_transfer_theory = get_filter_spec(
            self.transfer_theory, lambda x: x.ctf.defocus_in_angstroms
        )
        transfer_vmap, transfer_novmap = eqx.partition(
            transfer_theory_batch, filter_spec_for_transfer_theory
        )
        contrast_spectrum_at_detector_plane = compute_image_superposition(
            (ensemble_vmap, transfer_vmap),
            (ensemble_novmap, transfer_novmap),
            instrument_config,
        )

        if rng_key is not None:
            # Get the contrast from the ice and add to that of the image batch
            if self.solvent is not None:
                fourier_ice_contrast_at_detector_plane = self.transfer_theory(
                    self.solvent.sample_ice_spectrum(rng_key, instrument_config),
                    instrument_config,
                )
                contrast_spectrum_at_detector_plane += (
                    fourier_ice_contrast_at_detector_plane
                )

        return contrast_spectrum_at_detector_plane


def _compute_object_spectrum_from_scattering_potential(
    structural_ensemble, potential_integrator, instrument_config
):
    # Get potential in the lab frame
    potential = structural_ensemble.get_potential_in_lab_frame()
    # Compute the phase shifts in the exit plane
    fourier_integrated_potential = (
        potential_integrator.compute_fourier_integrated_potential(
            potential, instrument_config
        )
    )
    # Compute the phase shifts in exit plane and multiply by the translation.
    phase_shifts_in_exit_plane = convert_units_of_integrated_potential(
        fourier_integrated_potential, instrument_config.wavelength_in_angstroms
    )
    return phase_shifts_in_exit_plane

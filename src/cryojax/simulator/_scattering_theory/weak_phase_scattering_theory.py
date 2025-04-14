from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...utils import batched_scan, get_filter_spec
from .._instrument_config import InstrumentConfig
from .._pose import AbstractPose
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractSolvent
from .._structural_ensemble import (
    AbstractAssembly,
    AbstractConformationalVariable,
    AbstractStructuralEnsemble,
)
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractScatteringTheory
from .common_functions import apply_interaction_constant


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
    solvent: Optional[AbstractSolvent] = None

    def __init__(
        self,
        structural_ensemble: AbstractStructuralEnsemble,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractSolvent] = None,
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

    @override
    def compute_object_spectrum_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # Compute the integrated potential
        fourier_integrated_potential = _integrate_potential_to_exit_plane(
            self.structural_ensemble, self.potential_integrator, instrument_config
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_integrated_potential = self.solvent.compute_integrated_potential_with_solvent(  # noqa: E501
                    rng_key,
                    fourier_integrated_potential,
                    instrument_config,
                    input_is_rfft=self.potential_integrator.is_projection_approximation,
                )

        object_spectrum_at_exit_plane = apply_interaction_constant(
            fourier_integrated_potential, instrument_config.wavelength_in_angstroms
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
        contrast_spectrum_at_detector_plane = self.transfer_theory.propagate_object_to_detector_plane(  # noqa: E501
            object_spectrum_at_exit_plane,
            instrument_config,
            is_projection_approximation=self.potential_integrator.is_projection_approximation,
        )
        # ... apply in-plane translation
        translational_phase_shifts = self.structural_ensemble.pose.compute_shifts(
            instrument_config.padded_frequency_grid_in_angstroms
        )
        contrast_spectrum_at_detector_plane *= translational_phase_shifts

        return contrast_spectrum_at_detector_plane


class LinearSuperpositionScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Compute the superposition of images over a batch of poses and potentials
    parameterized by an `AbstractAssembly`. This must operate in the weak phase
    approximation.
    """

    structural_ensemble: AbstractAssembly
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractSolvent]

    batch_size: int

    def __init__(
        self,
        structural_ensemble: AbstractAssembly,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractSolvent] = None,
        *,
        batch_size: int = 1,
    ):
        """**Arguments:**

        - `structural_ensemble`:
            An concrete class of an `AbstractAssembly`. This is used to
            output a batch of states over which to
            compute a superposition of images.
        - `potential_integrator`: The method for integrating the specimen potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        - `batch_size`: The number of images to compute in parallel with vmap.
        """
        self.structural_ensemble = structural_ensemble
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent
        self.batch_size = batch_size

    @override
    def compute_object_spectrum_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        @eqx.filter_vmap(in_axes=(0, None, None))
        def compute_image_stack(ensemble_vmap, ensemble_novmap, instrument_config):
            ensemble = eqx.combine(ensemble_vmap, ensemble_novmap)
            fourier_integrated_potential = _integrate_potential_to_exit_plane(
                ensemble, self.potential_integrator, instrument_config
            )
            return fourier_integrated_potential

        # Get the batch
        ensemble_batch, _ = (
            self.structural_ensemble.get_subcomponents_and_z_positions_in_lab_frame()
        )
        # Setup vmap over the pose and conformation
        is_vmap = lambda x: isinstance(x, (AbstractPose, AbstractConformationalVariable))
        to_vmap = jax.tree_util.tree_map(is_vmap, ensemble_batch, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(ensemble_batch, to_vmap)

        fourier_integrated_potential = _compute_image_superposition(
            vmap, novmap, instrument_config, compute_image_stack
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_integrated_potential = self.solvent.compute_integrated_potential_with_solvent(  # noqa: E501
                    rng_key,
                    fourier_integrated_potential,
                    instrument_config,
                    input_is_rfft=self.potential_integrator.is_projection_approximation,
                )

        object_spectrum_at_exit_plane = apply_interaction_constant(
            fourier_integrated_potential, instrument_config.wavelength_in_angstroms
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
        @eqx.filter_vmap(in_axes=(0, None, None))
        def compute_image_stack(pytree_vmap, pytree_novmap, instrument_config):
            ensemble_vmap, transfer_vmap = pytree_vmap
            ensemble_novmap, transfer_novmap = pytree_novmap
            ensemble = eqx.combine(ensemble_vmap, ensemble_novmap)
            transfer_theory = eqx.combine(transfer_vmap, transfer_novmap)
            fourier_integrated_potential = _integrate_potential_to_exit_plane(
                ensemble, self.potential_integrator, instrument_config
            )
            object_spectrum_at_exit_plane = apply_interaction_constant(
                fourier_integrated_potential, instrument_config.wavelength_in_angstroms
            )
            translational_phase_shifts = ensemble.pose.compute_shifts(
                instrument_config.padded_frequency_grid_in_angstroms
            )
            contrast_spectrum_at_detector_plane = transfer_theory.propagate_object_to_detector_plane(  # noqa: E501
                object_spectrum_at_exit_plane,
                instrument_config,
                is_projection_approximation=self.potential_integrator.is_projection_approximation,
            )

            return translational_phase_shifts * contrast_spectrum_at_detector_plane

        # Get the batches
        ensemble_batch, z_positions = (
            self.structural_ensemble.get_subcomponents_and_z_positions_in_lab_frame()
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
        contrast_spectrum_at_detector_plane = _compute_image_superposition(
            (ensemble_vmap, transfer_vmap),
            (ensemble_novmap, transfer_novmap),
            instrument_config,
            compute_image_stack,
        )

        if rng_key is not None:
            # Get the contrast from the ice and add to that of the image batch
            if self.solvent is not None:
                fourier_integrated_potential_of_solvent = (
                    self.solvent.sample_solvent_integrated_potential(
                        rng_key,
                        instrument_config,
                    )
                )
                solvent_spectrum_at_exit_plane = apply_interaction_constant(
                    fourier_integrated_potential_of_solvent,
                    instrument_config.wavelength_in_angstroms,
                )
                solvent_contrast_spectrum_at_detector_plane = (
                    self.transfer_theory.propagate_object_to_detector_plane(
                        solvent_spectrum_at_exit_plane,
                        instrument_config,
                        is_projection_approximation=True,
                    )
                )
                contrast_spectrum_at_detector_plane += (
                    solvent_contrast_spectrum_at_detector_plane
                )

        return contrast_spectrum_at_detector_plane


def _integrate_potential_to_exit_plane(
    structural_ensemble, potential_integrator, instrument_config
):
    # Get potential in the lab frame
    potential = structural_ensemble.get_potential_in_lab_frame()
    # Compute the phase shifts in the exit plane
    fourier_integrated_potential = potential_integrator.compute_integrated_potential(
        potential, instrument_config, outputs_real_space=False
    )
    return fourier_integrated_potential


@eqx.filter_jit
def _compute_image_superposition(
    pytree_vmap, pytree_novmap, instrument_config, compute_fn
):
    output_shape = (
        instrument_config.padded_y_dim,
        instrument_config.padded_x_dim // 2 + 1,
    )
    init = jnp.zeros(output_shape, dtype=complex)

    def f_scan(carry, xs):
        image_stack = compute_fn(xs, pytree_novmap, instrument_config)
        return carry.at[:, :].add(jnp.sum(image_stack, axis=0)), None

    image, _ = batched_scan(f_scan, init, pytree_vmap, batch_size=1)

    return image

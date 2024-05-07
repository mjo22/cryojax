from abc import abstractmethod
from functools import partial
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from .._instrument_config import InstrumentConfig
from .._pose import AbstractPose
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractIce
from .._structural_ensemble import (
    AbstractConformationalVariable,
    AbstractStructuralEnsemble,
    AbstractStructuralEnsembleBatcher,
)
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractScatteringTheory


class AbstractLinearScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a scattering theory in linear image formation theory
    (the weak-phase approximation).
    """

    @abstractmethod
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @override
    def compute_fourier_squared_wavefunction_at_detector_plane(
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
        fourier_contrast_at_detector_plane = (
            self.compute_fourier_contrast_at_detector_plane(instrument_config, rng_key)
        )
        fourier_squared_wavefunction_at_detector_plane = (
            (2 * fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
        )
        return fourier_squared_wavefunction_at_detector_plane


class LinearScatteringTheory(AbstractLinearScatteringTheory, strict=True):
    """Base linear image formation theory."""

    structural_ensemble: AbstractStructuralEnsemble
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce] = None

    @override
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # Compute the phase shifts in the exit plane
        fourier_phase_shifts_at_exit_plane = (
            _compute_phase_shifts_from_projected_potential(
                self.structural_ensemble, self.potential_integrator, instrument_config
            )
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_phase_shifts_at_exit_plane = (
                    self.solvent.compute_fourier_phase_shifts_with_ice(
                        rng_key, fourier_phase_shifts_at_exit_plane, instrument_config
                    )
                )

        return fourier_phase_shifts_at_exit_plane

    @override
    def compute_fourier_contrast_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        fourier_phase_shifts_at_exit_plane = (
            self.compute_fourier_phase_shifts_at_exit_plane(instrument_config, rng_key)
        )
        fourier_contrast_at_detector_plane = self.transfer_theory(
            fourier_phase_shifts_at_exit_plane,
            instrument_config,
            defocus_offset=self.structural_ensemble.pose.offset_z_in_angstroms,
        )

        return fourier_contrast_at_detector_plane


LinearScatteringTheory.__init__.__doc__ = """**Arguments:**

- `structural_ensemble`: The structural ensemble of scattering potentials.
- `potential_integrator`: The method for integrating the scattering potential.
- `transfer_theory`: The contrast transfer theory.
- `solvent`: The model for the solvent.
"""


class LinearSuperpositionScatteringTheory(AbstractLinearScatteringTheory, strict=True):
    """Compute the superposition of images of the structural ensemble batch returned by
    the `AbstractStructuralEnsembleBatcher`.
    """

    structural_ensemble_batcher: AbstractStructuralEnsembleBatcher
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce] = None

    @override
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        @partial(eqx.filter_vmap, in_axes=(0, None, None))
        def compute_image_stack(ensemble_vmap, ensemble_no_vmap, instrument_config):
            ensemble = eqx.combine(ensemble_vmap, ensemble_no_vmap)
            fourier_phase_shifts_at_exit_plane = (
                _compute_phase_shifts_from_projected_potential(
                    ensemble, self.potential_integrator, instrument_config
                )
            )
            return fourier_phase_shifts_at_exit_plane

        @eqx.filter_jit
        def compute_image_superposition(
            ensemble_vmap, ensemble_no_vmap, instrument_config
        ):
            return jnp.sum(
                compute_image_stack(ensemble_vmap, ensemble_no_vmap, instrument_config),
                axis=0,
            )

        # Get the batch
        ensemble_batch = (
            self.structural_ensemble_batcher.get_batched_structural_ensemble()
        )
        # Setup vmap over the pose and conformation
        is_vmap = lambda x: isinstance(x, (AbstractPose, AbstractConformationalVariable))
        to_vmap = jax.tree_util.tree_map(is_vmap, ensemble_batch, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(ensemble_batch, to_vmap)

        fourier_phase_shifts_at_exit_plane = compute_image_superposition(
            vmap, novmap, instrument_config
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_phase_shifts_at_exit_plane = (
                    self.solvent.compute_fourier_phase_shifts_with_ice(
                        rng_key, fourier_phase_shifts_at_exit_plane, instrument_config
                    )
                )

        return fourier_phase_shifts_at_exit_plane

    @override
    def compute_fourier_contrast_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        @partial(eqx.filter_vmap, in_axes=(0, None, None))
        def compute_image_stack(ensemble_vmap, ensemble_no_vmap, instrument_config):
            ensemble = eqx.combine(ensemble_vmap, ensemble_no_vmap)
            fourier_phase_shifts_at_exit_plane = (
                _compute_phase_shifts_from_projected_potential(
                    ensemble, self.potential_integrator, instrument_config
                )
            )
            fourier_contrast_at_detector_plane = self.transfer_theory(
                fourier_phase_shifts_at_exit_plane, instrument_config
            )

            return fourier_contrast_at_detector_plane

        @eqx.filter_jit
        def compute_image_superposition(
            ensemble_vmap, ensemble_no_vmap, instrument_config
        ):
            return jnp.sum(
                compute_image_stack(ensemble_vmap, ensemble_no_vmap, instrument_config),
                axis=0,
            )

        # Get the batch
        ensemble_batch = (
            self.structural_ensemble_batcher.get_batched_structural_ensemble()
        )
        # Setup vmap over the pose and conformation
        is_vmap = lambda x: isinstance(x, (AbstractPose, AbstractConformationalVariable))
        to_vmap = jax.tree_util.tree_map(is_vmap, ensemble_batch, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(ensemble_batch, to_vmap)

        fourier_contrast_at_detector_plane = compute_image_superposition(
            vmap, novmap, instrument_config
        )

        if rng_key is not None:
            # Get the contrast from the ice and add to that of the image batch
            if self.solvent is not None:
                fourier_ice_contrast_at_detector_plane = self.transfer_theory(
                    self.solvent.sample_fourier_phase_shifts_from_ice(
                        rng_key, instrument_config
                    ),
                    instrument_config,
                )
                fourier_contrast_at_detector_plane += (
                    fourier_ice_contrast_at_detector_plane
                )

        return fourier_contrast_at_detector_plane


LinearSuperpositionScatteringTheory.__init__.__doc__ = """**Arguments:**

- `structural_ensemble_batcher`: The batcher that computes the states that over which to
                                 compute a superposition of images. Most commonly, this
                                 would be an `AbstractAssembly` concrete class.
- `potential_integrator`: The method for integrating the specimen potential.
- `transfer_theory`: The contrast transfer theory.
- `solvent`: The model for the solvent.
"""


def _compute_phase_shifts_from_projected_potential(
    structural_ensemble, potential_integrator, instrument_config
):
    # Get potential in the lab frame
    potential = structural_ensemble.get_potential_in_lab_frame()
    # Compute the phase shifts in the exit plane
    fourier_projected_potential = (
        potential_integrator.compute_fourier_integrated_potential(
            potential, instrument_config
        )
    )
    # Compute in-plane translation through fourier phase shifts
    translational_phase_shifts = structural_ensemble.pose.compute_shifts(
        instrument_config.wrapped_padded_frequency_grid_in_angstroms.get()
    )
    # The phase shifts in the exit plane multiplies the wavelength x
    # projected potential (here with units of inverse angstroms) x the translation
    return (
        instrument_config.wavelength_in_angstroms
        * fourier_projected_potential
        * translational_phase_shifts
    )

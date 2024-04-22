from abc import abstractmethod
from functools import partial
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from .._assembly import AbstractAssembly
from .._config import ImageConfig
from .._ensemble import AbstractConformation, AbstractPotentialEnsemble
from .._ice import AbstractIce
from .._instrument import Instrument
from .._pose import AbstractPose
from .._projection_methods import AbstractPotentialProjectionMethod
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractScatteringTheory


class AbstractLinearScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a scattering theory in linear image formation theory."""

    projection_method: eqx.AbstractVar[AbstractPotentialProjectionMethod]
    transfer_theory: eqx.AbstractVar[ContrastTransferTheory]

    @abstractmethod
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @abstractmethod
    def compute_fourier_contrast_at_detector_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @override
    def compute_fourier_squared_wavefunction_at_detector_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = config.padded_shape
        # ... compute the squared wavefunction directly from the image contrast
        # as |psi|^2 = 1 + 2C.
        fourier_contrast_at_detector_plane = (
            self.compute_fourier_contrast_at_detector_plane(config, instrument, rng_key)
        )
        fourier_squared_wavefunction_at_detector_plane = (
            (2 * fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
        )
        return fourier_squared_wavefunction_at_detector_plane


class LinearScatteringTheory(AbstractLinearScatteringTheory, strict=True):
    """Base linear image formation theory."""

    potential_ensemble: AbstractPotentialEnsemble
    projection_method: AbstractPotentialProjectionMethod
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce] = None

    @override
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        # Get potential in the lab frame
        potential = self.potential_ensemble.get_potential_in_lab_frame()
        # Compute the phase shifts in the exit plane
        fourier_projected_potential = (
            self.projection_method.compute_fourier_projected_potential(
                potential, config
            )
        )
        fourier_phase_at_exit_plane = (
            instrument.wavelength_in_angstroms * fourier_projected_potential
        )
        # Apply in-plane translation through phase shifts
        fourier_phase_at_exit_plane *= self.potential_ensemble.pose.compute_shifts(
            config.wrapped_padded_frequency_grid_in_angstroms.get()
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_phase_at_exit_plane = self.solvent(
                    rng_key, fourier_phase_at_exit_plane, config
                )

        return fourier_phase_at_exit_plane

    @override
    def compute_fourier_contrast_at_detector_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        fourier_phase_at_exit_plane = self.compute_fourier_phase_shifts_at_exit_plane(
            config, instrument, rng_key
        )
        fourier_contrast_at_detector_plane = self.transfer_theory(
            fourier_phase_at_exit_plane,
            config,
            instrument.wavelength_in_angstroms,
            defocus_offset=self.potential_ensemble.pose.offset_z_in_angstroms,
        )

        return fourier_contrast_at_detector_plane


LinearScatteringTheory.__init__.__doc__ = """**Arguments:**

- `potential_ensemble`: The specimen potential ensemble.
- `projection_method`: The method for computing projections of the specimen potential.
- `transfer_theory`: The contrast transfer theory.
- `solvent`: The model for the solvent.
"""


class LinearSuperpositionScatteringTheory(AbstractLinearScatteringTheory, strict=True):
    """Compute the superposition of images stored in `AbstractAssembly.subunits`."""

    assembly: AbstractAssembly
    projection_method: AbstractPotentialProjectionMethod
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractIce]

    @override
    def compute_fourier_phase_shifts_at_exit_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        @partial(eqx.filter_vmap, in_axes=(0, None, None, None))
        def compute_subunit_stack(ensemble_vmap, ensemble_no_vmap, instrument, config):
            ensemble = eqx.combine(ensemble_vmap, ensemble_no_vmap)
            # Get potential in the lab frame
            potential = ensemble.get_potential_in_lab_frame()
            # Compute the phase shifts in the exit plane
            fourier_projected_potential = (
                self.projection_method.compute_fourier_projected_potential(
                    potential, config
                )
            )
            fourier_phase_at_exit_plane = (
                instrument.wavelength_in_angstroms * fourier_projected_potential
            )
            # Apply in-plane translation through phase shifts
            fourier_phase_at_exit_plane *= ensemble.pose.compute_shifts(
                config.wrapped_padded_frequency_grid_in_pixels.get()
            )

            return fourier_phase_at_exit_plane

        @eqx.filter_jit
        def compute_subunit_superposition(
            ensemble_vmap, ensemble_no_vmap, instrument, config
        ):
            return jnp.sum(
                compute_subunit_stack(
                    ensemble_vmap, ensemble_no_vmap, instrument, config
                ),
                axis=0,
            )

        # Get the assembly subunits
        subunits = self.assembly.subunits
        # Setup vmap over the pose and conformation
        is_vmap = lambda x: isinstance(x, (AbstractPose, AbstractConformation))
        to_vmap = jax.tree_util.tree_map(is_vmap, subunits, is_leaf=is_vmap)
        vmap, novmap = eqx.partition(subunits, to_vmap)

        fourier_phase_at_exit_plane = compute_subunit_superposition(
            vmap, novmap, instrument, config
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_phase_at_exit_plane = self.solvent(
                    rng_key, fourier_phase_at_exit_plane, config
                )

        return fourier_phase_at_exit_plane

    @override
    def compute_fourier_contrast_at_detector_plane(
        self,
        config: ImageConfig,
        instrument: Instrument,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        fourier_phase_at_exit_plane = self.compute_fourier_phase_shifts_at_exit_plane(
            config, instrument, rng_key
        )
        fourier_contrast_at_detector_plane = self.transfer_theory(
            fourier_phase_at_exit_plane,
            config,
            instrument.wavelength_in_angstroms,
            defocus_offset=self.assembly.pose.offset_z_in_angstroms,
        )

        return fourier_contrast_at_detector_plane


LinearScatteringTheory.__init__.__doc__ = """**Arguments:**

- `assembly`: The assembly of subunits over which to compute a superposition of images.
- `projection_method`: The method for computing projections of the specimen potential.
- `transfer_theory`: The contrast transfer theory.
- `solvent`: The model for the solvent.
"""

from abc import abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ._config import ImageConfig
from ._ice import AbstractIce
from ._optics import ContrastTransferTheory
from ._specimen import AbstractSpecimen


class AbstractScatteringTheory(eqx.Module, strict=True):
    @abstractmethod
    def scatter_to_focal_plane(self, config: ImageConfig) -> Array:
        raise NotImplementedError

    @abstractmethod
    def scatter_to_focal_plane_with_solvent(
        self, key: PRNGKeyArray, config: ImageConfig
    ) -> Array:
        raise NotImplementedError


class FarFieldScatteringTheory(AbstractScatteringTheory, strict=True):
    specimen: AbstractSpecimen
    optics: ContrastTransferTheory
    solvent: Optional[AbstractIce]

    def scatter_to_exit_plane(
        self,
        config: ImageConfig,
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Scatter the specimen potential to the exit plane."""
        # Get potential in the lab frame
        potential = self.potential_in_lab_frame
        # Compute the scattering potential in fourier space
        fourier_phase_at_exit_plane = self.integrator(
            potential, wavelength_in_angstroms, config
        )
        # Apply translation through phase shifts
        fourier_phase_at_exit_plane *= self.pose.compute_shifts(
            config.wrapped_padded_frequency_grid_in_angstroms.get()
        )

        return fourier_phase_at_exit_plane

    def scatter_to_exit_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        config: ImageConfig,
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Scatter the specimen potential to the exit plane, including
        the phase shifts due to the solvent."""
        # Compute the phase  in fourier space
        fourier_phase_at_exit_plane = self.scatter_to_exit_plane(
            config, wavelength_in_angstroms
        )
        # Get the potential of the specimen plus the ice
        fourier_phase_at_exit_plane_with_solvent = self.solvent(
            key, fourier_phase_at_exit_plane, config
        )

        return fourier_phase_at_exit_plane_with_solvent

    def propagate_to_detector_plane(
        self,
        fourier_phase_at_exit_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> (
        Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
        | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        if self.optics is None:
            raise AttributeError(
                "Tried to call `Instrument.propagate_to_detector_plane`, "
                "but the `Instrument`'s optics model is `None`. This "
                "is not allowed!"
            )
        """Propagate the scattering potential with the optics model."""
        fourier_contrast_at_detector_plane = self.optics(
            fourier_phase_at_exit_plane,
            config,
            self.wavelength_in_angstroms,
            defocus_offset=defocus_offset,
        )

        return fourier_contrast_at_detector_plane

    def compute_fourier_squared_wavefunction(
        self,
        fourier_contrast_at_detector_plane: (
            Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
            | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        ),
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = config.padded_shape
        if self.optics is None:
            raise AttributeError(
                "Tried to call `compute_fourier_squared_wavefunction`, "
                "but the `Instrument`'s optics model is `None`. This "
                "is not allowed!"
            )
        elif self.optics.is_linear:
            # ... compute the squared wavefunction directly from the image contrast
            # as |psi|^2 = 1 + 2C.
            fourier_contrast_at_detector_plane = fourier_contrast_at_detector_plane
            fourier_squared_wavefunction_at_detector_plane = (
                (2 * fourier_contrast_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
            )
            return fourier_squared_wavefunction_at_detector_plane
        else:
            raise NotImplementedError(
                "Functionality for AbstractOptics.is_linear = False not supported."
            )

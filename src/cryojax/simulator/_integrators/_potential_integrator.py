"""
Base class for a method of integration of the scattering potential onto the exit plane.
"""

from abc import abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from equinox import Module, AbstractVar

from .._specimen import AbstractSpecimen
from .._potential import AbstractScatteringPotential, AbstractVoxelPotential
from .._config import ImageConfig
from .._ice import AbstractIce

from ...image import rfftn, irfftn
from ...typing import ComplexImage


class AbstractPotentialIntegrator(Module, strict=True):
    """Base class for a method to integrate the potential onto
    the exit plane."""

    config: AbstractVar[ImageConfig]

    @abstractmethod
    def integrate_potential(
        self, potential: AbstractScatteringPotential
    ) -> ComplexImage:
        """Compute the scattering potential in the exit plane.

        **Arguments:**

        `potential`: The scattering potential representation.
        """
        raise NotImplementedError

    def scatter_to_exit_plane(self, specimen: AbstractSpecimen) -> ComplexImage:
        """Scatter the specimen potential to the exit plane."""
        # Get potential in the lab frame
        potential = specimen.potential_in_lab_frame
        # Compute the fourier projection in the exit plane
        fourier_potential_at_exit_plane = self.integrate_potential(potential)
        # Rescale the pixel size if different from the voxel size
        if isinstance(potential, AbstractVoxelPotential):
            rescale_fn = lambda fourier_potential: rfftn(
                self.config.rescale_to_pixel_size(
                    irfftn(fourier_potential, s=self.config.padded_shape),
                    potential.voxel_size,
                )
            )
            null_fn = lambda fourier_potential: fourier_potential
            fourier_potential_at_exit_plane = jax.lax.cond(
                jnp.isclose(potential.voxel_size, self.config.pixel_size),
                null_fn,
                rescale_fn,
                fourier_potential_at_exit_plane,
            )
        # Apply translation through phase shifts
        fourier_potential_at_exit_plane *= specimen.pose.compute_shifts(
            self.config.padded_frequency_grid_in_angstroms.get()
        )

        return fourier_potential_at_exit_plane

    def scatter_to_exit_plane_with_solvent(
        self, key: PRNGKeyArray, specimen: AbstractSpecimen, solvent: AbstractIce
    ) -> ComplexImage:
        """Scatter the specimen potential to the exit plane, including
        the potential due to the solvent."""
        # Compute the scattering potential in fourier space
        fourier_potential_at_exit_plane = self.scatter_to_exit_plane(specimen)
        # Get the potential of the specimen plus the ice
        fourier_potential_at_exit_plane_with_solvent = solvent(
            key, fourier_potential_at_exit_plane, self.config
        )

        return fourier_potential_at_exit_plane_with_solvent

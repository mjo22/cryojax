"""
Abstractions of biological specimen.
"""

from abc import abstractmethod
from functools import cached_property
from typing import Any, Optional
from typing_extensions import override

from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, PRNGKeyArray

from .._config import ImageConfig
from .._ice import AbstractIce
from .._instrument import Instrument
from .._integrators import AbstractPotentialIntegrator
from .._pose import AbstractPose, EulerAnglePose
from .._potential import AbstractScatteringPotential


class AbstractSpecimen(Module, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `potential`: The scattering potential of the specimen.
    - `integrator`: A method of integrating the `potential` onto the exit
                    plane of the specimen.
    - `pose`: The pose of the specimen.
    """

    potential: AbstractVar[Any]
    integrator: AbstractVar[Any]
    pose: AbstractVar[AbstractPose]

    @cached_property
    @abstractmethod
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the center of mass
        frame."""
        raise NotImplementedError

    @cached_property
    def potential_in_lab_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the lab frame."""
        return self.potential_in_com_frame.rotate_to_pose(self.pose)

    def scatter_to_exit_plane(
        self,
        instrument: Instrument,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Scatter the specimen potential to the exit plane."""
        # Get potential in the lab frame
        potential = self.potential_in_lab_frame
        # Compute the scattering potential in fourier space
        fourier_phase_at_exit_plane = self.integrator(
            potential, instrument.wavelength_in_angstroms, config
        )
        # Apply translation through phase shifts
        fourier_phase_at_exit_plane *= self.pose.compute_shifts(
            config.wrapped_padded_frequency_grid_in_angstroms.get()
        )

        return fourier_phase_at_exit_plane

    def scatter_to_exit_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        instrument: Instrument,
        solvent: AbstractIce,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Scatter the specimen potential to the exit plane, including
        the phase shifts due to the solvent."""
        # Compute the phase  in fourier space
        fourier_phase_at_exit_plane = self.scatter_to_exit_plane(instrument, config)
        # Get the potential of the specimen plus the ice
        fourier_phase_at_exit_plane_with_solvent = solvent(
            key, fourier_phase_at_exit_plane, config
        )

        return fourier_phase_at_exit_plane_with_solvent


class Specimen(AbstractSpecimen, strict=True):
    """
    Abstraction of a of biological specimen.

    **Attributes:**

    - `potential`: The scattering potential representation of the
                    specimen as a single scattering potential object.
    """

    potential: AbstractScatteringPotential
    integrator: AbstractPotentialIntegrator
    pose: AbstractPose

    def __init__(
        self,
        potential: AbstractScatteringPotential,
        integrator: AbstractPotentialIntegrator,
        pose: Optional[AbstractPose] = None,
    ):
        self.potential = potential
        self.integrator = integrator
        self.pose = pose or EulerAnglePose()

    @cached_property
    @override
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential in the center of mass
        frame."""
        return self.potential

"""
Abstractions of biological specimen.
"""

from abc import abstractmethod
from functools import cached_property
from typing import Any, Optional
from typing_extensions import override

import jax
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, PRNGKeyArray

from ._config import ImageConfig
from ._conformation import AbstractConformation, DiscreteConformation
from ._ice import AbstractIce
from ._instrument import Instrument
from ._integrators import AbstractPotentialIntegrator
from ._pose import AbstractPose, EulerAnglePose
from ._potential import AbstractScatteringPotential


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
        fourier_potential_at_exit_plane = self.integrator(
            potential, instrument.wavelength_in_angstroms, config
        )
        # Apply translation through phase shifts
        fourier_potential_at_exit_plane *= self.pose.compute_shifts(
            config.wrapped_padded_frequency_grid_in_angstroms.get()
        )

        return fourier_potential_at_exit_plane

    def scatter_to_exit_plane_with_solvent(
        self,
        key: PRNGKeyArray,
        instrument: Instrument,
        solvent: AbstractIce,
        config: ImageConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Scatter the specimen potential to the exit plane, including
        the potential due to the solvent."""
        # Compute the scattering potential in fourier space
        fourier_potential_at_exit_plane = self.scatter_to_exit_plane(instrument, config)
        # Get the potential of the specimen plus the ice
        fourier_potential_at_exit_plane_with_solvent = solvent(
            key, fourier_potential_at_exit_plane, config
        )

        return fourier_potential_at_exit_plane_with_solvent


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


class AbstractEnsemble(AbstractSpecimen, strict=True):
    """
    Abstraction of an ensemble of a biological specimen which can
    occupy different conformations.

    **Attributes:**

    - `conformation`: The conformation at which to evaluate the scattering potential.
    """

    conformation: AbstractVar[AbstractConformation]


class DiscreteEnsemble(AbstractEnsemble, strict=True):
    """
    Abstraction of an ensemble with discrete conformational
    heterogeneity.

    **Attributes:**

    - `potential`: A tuple of scattering potential representations.
    - `pose`: The pose of the specimen.
    - `conformation`: A conformation with a discrete index at which to evaluate
                      the scattering potential tuple.
    """

    potential: tuple[AbstractScatteringPotential, ...]
    integrator: AbstractPotentialIntegrator
    pose: AbstractPose
    conformation: DiscreteConformation

    def __init__(
        self,
        potential: tuple[AbstractScatteringPotential, ...],
        integrator: AbstractPotentialIntegrator,
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformation] = None,
    ):
        self.potential = potential
        self.integrator = integrator
        self.pose = pose or EulerAnglePose()
        self.conformation = conformation or DiscreteConformation(0)

    @cached_property
    @override
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential at configured conformation."""
        funcs = [lambda i=i: self.potential[i] for i in range(len(self.potential))]
        potential = jax.lax.switch(self.conformation.value, funcs)

        return potential

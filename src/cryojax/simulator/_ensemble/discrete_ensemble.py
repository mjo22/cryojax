"""
Abstractions of ensembles on discrete conformational variables.
"""

from typing import Optional
from typing_extensions import override

import jax
from equinox import field
from jaxtyping import Array, Int

from ..._errors import error_if_negative
from .._pose import AbstractPose, EulerAnglePose
from .._potential import AbstractPotentialRepresentation
from .base_conformation import AbstractConformationalVariable
from .base_ensemble import AbstractStructuralEnsemble


class DiscreteConformationalVariable(AbstractConformationalVariable, strict=True):
    """A conformational variable as a discrete index."""

    value: Int[Array, ""] = field(converter=error_if_negative)


class DiscreteStructuralEnsemble(AbstractStructuralEnsemble, strict=True):
    """Abstraction of an ensemble with discrete conformational
    heterogeneity.
    """

    conformational_space: tuple[AbstractPotentialRepresentation, ...]
    pose: AbstractPose
    conformation: DiscreteConformationalVariable

    def __init__(
        self,
        conformational_space: tuple[AbstractPotentialRepresentation, ...],
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformationalVariable] = None,
    ):
        """**Arguments:**

        - `conformational_space`: A tuple of `AbstractPotential` representations.
        - `pose`: The pose of the specimen.
        - `conformation`: A conformation with a discrete index at which to evaluate
                          the scattering potential tuple.
        """
        self.conformational_space = conformational_space
        self.pose = pose or EulerAnglePose()
        self.conformation = conformation or DiscreteConformationalVariable(0)

    @override
    def get_potential_at_conformation(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential at configured conformation."""
        funcs = [
            lambda i=i: self.conformational_space[i]
            for i in range(len(self.conformational_space))
        ]
        potential = jax.lax.switch(self.conformation.value, funcs)

        return potential

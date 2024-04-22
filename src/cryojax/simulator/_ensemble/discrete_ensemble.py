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
from .._potential import AbstractSpecimenPotential
from .base_conformation import AbstractConformation
from .base_ensemble import AbstractPotentialEnsemble


class DiscreteConformation(AbstractConformation, strict=True):
    """
    A conformational variable wrapped in a Module.
    """

    value: Int[Array, ""] = field(converter=error_if_negative)


class DiscreteEnsemble(AbstractPotentialEnsemble, strict=True):
    """
    Abstraction of an ensemble with discrete conformational
    heterogeneity.
    """

    state_space: tuple[AbstractSpecimenPotential, ...]
    pose: AbstractPose
    conformation: DiscreteConformation

    def __init__(
        self,
        state_space: tuple[AbstractSpecimenPotential, ...],
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformation] = None,
    ):
        """**Arguments:**

        - `state_space`: A tuple of specimen potential representations.
        - `pose`: The pose of the specimen.
        - `conformation`: A conformation with a discrete index at which to evaluate
                          the scattering potential tuple.
        """
        self.state_space = state_space
        self.pose = pose or EulerAnglePose()
        self.conformation = conformation or DiscreteConformation(0)

    @override
    def get_potential(self) -> AbstractSpecimenPotential:
        """Get the scattering potential at configured conformation."""
        funcs = [lambda i=i: self.state_space[i] for i in range(len(self.state_space))]
        potential = jax.lax.switch(self.conformation.value, funcs)

        return potential

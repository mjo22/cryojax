"""
Abstractions of ensembles of biological specimen.
"""

from functools import cached_property
from typing import Optional
from typing_extensions import override

import jax
from equinox import AbstractVar

from .._pose import AbstractPose, EulerAnglePose
from .._potential import AbstractScatteringPotential
from .conformation import AbstractConformation, DiscreteConformation
from .specimen import AbstractSpecimen


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
    pose: AbstractPose
    conformation: DiscreteConformation

    def __init__(
        self,
        potential: tuple[AbstractScatteringPotential, ...],
        pose: Optional[AbstractPose] = None,
        conformation: Optional[DiscreteConformation] = None,
    ):
        self.potential = potential
        self.pose = pose or EulerAnglePose()
        self.conformation = conformation or DiscreteConformation(0)

    @cached_property
    @override
    def potential_in_com_frame(self) -> AbstractScatteringPotential:
        """Get the scattering potential at configured conformation."""
        funcs = [lambda i=i: self.potential[i] for i in range(len(self.potential))]
        potential = jax.lax.switch(self.conformation.value, funcs)

        return potential

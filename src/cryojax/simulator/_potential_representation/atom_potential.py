"""
from typing import Any, ClassVar, Type

import equinox as eqx
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array

from .._pose import AbstractPose
from .scattering_potential import AbstractScatteringPotential


class AtomCloud(AbstractScatteringPotential):
    '''
    Abstraction of a point cloud of atoms.
    '''

    weights: Array = field(converter=jnp.asarray)
    coordinate_list: Array = field(converter=jnp.asarray)
    variances: Array = field(converter=jnp.asarray)
    identity: Array = field(converter=jnp.asarray)

    is_real: ClassVar[bool] = True

    def rotate_to(self, pose: AbstractPose) -> "AtomCloud":
        return eqx.tree_at(
            lambda d: d.coordinate_list,
            self,
            pose.rotate_coordinates(self.coordinate_list, is_real=self.is_real),
        )

    @classmethod
    def from_file(
        cls: Type["AtomCloud"],
        filename: str,
        **kwargs: Any,
    ) -> "AtomCloud":
        '''
        Load an Atom Cloud

        TODO: What is the file format appropriate here? Q. for Michael...
        '''

        raise NotImplementedError
        # return cls.from_mrc(filename, config=config, **kwargs)
"""

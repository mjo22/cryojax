
from ..density import Ellipsoid
from ._scattering import ScatteringConfig
from ..pose import Pose
import jax.numpy as jnp
from typing import Union



class ShapeProjection(ScatteringConfig):

    def scatter(self,
                density: Ellipsoid, # add more shapes later with Union
                pose: Pose,
                ):
        # TODO: Implement this method
        n_pix = 1
        density.axis_1, density.axis_2, density.axis_3
        pose.rotation.as_matrix()
        return jnp.zeros((n_pix,n_pix))
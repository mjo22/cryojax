
from ..density import Ellipsoid
from ._scattering import ScatteringConfig
from ..pose import Pose
import jax.numpy as jnp


class ProjectEllipsoid(ScatteringConfig):

    def scatter(self,
                density: Ellipsoid,
                pose: Pose,
                ):
        # TODO: Implement this method
        n_pix = 1
        density.axis_1, density.axis_2, density.axis_3
        return jnp.zeros((n_pix,n_pix))
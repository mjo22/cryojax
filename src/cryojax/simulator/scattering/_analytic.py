
from ..density import Ellipsoid
from ._scattering import ScatteringConfig
from ..pose import Pose
import jax.numpy as jnp
from typing import Union
from ...typing import Real_
from ...utils import fftn


__all__ = ["ShapeProjection"]


class ShapeProjection(ScatteringConfig):

    def scatter(self,
                density: Ellipsoid, # add more shapes later with Union
                pose: Pose,
                resolution: Real_,
                ):
        # TODO: Implement this method
        n_pix = 3
        a,b,c = density.axis_a, density.axis_b, density.axis_c
        rotation = pose.rotation.as_matrix()
        # nx,ny = self.padded_shape
        physical_coords = self.padded_coords*resolution
        x_mesh, y_mesh = physical_coords[...,0], physical_coords[...,1] # whatever jax transpose is
        Rxx,Rxy,Rxz = rotation[0]
        Ryx,Ryy,Ryz = rotation[1]
        Rzx,Rzy,Rzz = rotation[2]
        x,y = x_mesh, y_mesh
        piece_2 = a*b*c*jnp.sqrt(-Rxx**2*Ryz**2*c**2*x**2 - Rxx**2*Rzz**2*b**2*x**2 - 2*Rxx*Rxy*Ryz**2*c**2*x*y - 2*Rxx*Rxy*Rzz**2*b**2*x*y + 2*Rxx*Rxz*Ryx*Ryz*c**2*x**2 + 2*Rxx*Rxz*Ryy*Ryz*c**2*x*y + 2*Rxx*Rxz*Rzx*Rzz*b**2*x**2 + 2*Rxx*Rxz*Rzy*Rzz*b**2*x*y - Rxy**2*Ryz**2*c**2*y**2 - Rxy**2*Rzz**2*b**2*y**2 + 2*Rxy*Rxz*Ryx*Ryz*c**2*x*y + 2*Rxy*Rxz*Ryy*Ryz*c**2*y**2 + 2*Rxy*Rxz*Rzx*Rzz*b**2*x*y + 2*Rxy*Rxz*Rzy*Rzz*b**2*y**2 - Rxz**2*Ryx**2*c**2*x**2 - 2*Rxz**2*Ryx*Ryy*c**2*x*y - Rxz**2*Ryy**2*c**2*y**2 - Rxz**2*Rzx**2*b**2*x**2 - 2*Rxz**2*Rzx*Rzy*b**2*x*y - Rxz**2*Rzy**2*b**2*y**2 + Rxz**2*b**2*c**2 - Ryx**2*Rzz**2*a**2*x**2 - 2*Ryx*Ryy*Rzz**2*a**2*x*y + 2*Ryx*Ryz*Rzx*Rzz*a**2*x**2 + 2*Ryx*Ryz*Rzy*Rzz*a**2*x*y - Ryy**2*Rzz**2*a**2*y**2 + 2*Ryy*Ryz*Rzx*Rzz*a**2*x*y + 2*Ryy*Ryz*Rzy*Rzz*a**2*y**2 - Ryz**2*Rzx**2*a**2*x**2 - 2*Ryz**2*Rzx*Rzy*a**2*x*y - Ryz**2*Rzy**2*a**2*y**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
        piece_3 = (Rxz**2*b**2*c**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
        z_nans = 2*piece_2/piece_3
        proj_ellipsoid = jnp.nan_to_num(z_nans, 0)
        return fftn(proj_ellipsoid)
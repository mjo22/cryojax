"""
Representations of rigid body rotations and translations of 3D coordinate systems.
"""

from abc import abstractmethod
from functools import cached_property
from typing_extensions import override, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float

from ..rotations import convert_quaternion_to_euler_angles, SO3


class AbstractPose(Module, strict=True):
    """Base class for the image pose. Subclasses will choose a
    particular convention for parameterizing the rotation by
    overwriting the `AbstractPose.rotation` property.

    !!! info
        Angular quantities in `cryojax` are always in *degrees*.
        Therefore concrete classes of the `AbstractPose` have
        angles in degrees, e.g.

        ```python
        import cryojax.simulator as cxs

        phi_in_degrees, theta_in_degrees, psi_in_degrees = 10.0, 30.0, 40.0
        pose = cxs.EulerAnglePose(
            view_phi=phi_in_degrees, view_theta=theta_in_degrees, view_psi=psi_in_degrees
        )
        ```
    """

    offset_x_in_angstroms: AbstractVar[Float[Array, ""]]
    offset_y_in_angstroms: AbstractVar[Float[Array, ""]]

    def rotate_coordinates(
        self,
        coordinate_grid_or_list: (
            Float[Array, "z_dim y_dim x_dim 3"] | Float[Array, "size 3"]
        ),
        inverse: bool = False,
    ) -> Float[Array, "z_dim y_dim x_dim 3"] | Float[Array, "size 3"]:
        """Rotate a 3D coordinate system.

        **Arguments:**

        - `coordinate_grid_or_list`:
            The 3D coordinate system to rotate. This can either be a list of coordinates
            of shape `(N, 3)` or a grid of coordinates `(N1, N2, N3, 3)`.
        - `inverse`:
            If `True`, compute the inverse rotation (i.e. rotation by the matrix $R^T$,
            where $R$ is the rotation matrix).

        **Returns:**

        The rotated version of `coordinate_grid_or_list`.
        """
        rotation = self.rotation.inverse() if inverse else self.rotation
        if isinstance(coordinate_grid_or_list, Float[Array, "size 3"]):  # type: ignore
            rotated_coordinate_grid_or_list = jax.vmap(rotation.apply)(
                coordinate_grid_or_list
            )
        elif isinstance(coordinate_grid_or_list, Float[Array, "z_dim y_dim x_dim 3"]):  # type: ignore
            rotated_coordinate_grid_or_list = jax.vmap(
                jax.vmap(jax.vmap(rotation.apply))
            )(coordinate_grid_or_list)
        else:
            raise ValueError(
                "Coordinates must be a JAX array either of shape (N, 3) or "
                f"(N1, N2, N3, 3). Instead, got {coordinate_grid_or_list.shape} and type "
                f"{type(coordinate_grid_or_list)}."
            )
        return rotated_coordinate_grid_or_list

    def compute_shifts(
        self, frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Compute the phase shifts from the in-plane translation,
        given a frequency grid coordinate system.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            A grid of in-plane frequency coordinates $(q_x, q_y)$

        **Returns:**

        From the vector $(t_x, t_y)$ (given by `self.offset_in_angstroms`), returns the
        grid of in-plane phase shifts $\\exp{(- 2 \\pi i (t_x q_x + t_y q_y))}$.
        """
        xy = self.offset_in_angstroms[0:2]
        return jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid_in_angstroms, xy)))

    @cached_property
    def offset_in_angstroms(self) -> Float[Array, "2"]:
        """The in-plane translation vector, where the origin in taken to
        be in the center of the imaging plane.
        """
        return jnp.asarray(
            (
                self.offset_x_in_angstroms,
                self.offset_y_in_angstroms,
            )
        )

    @cached_property
    @abstractmethod
    def rotation(self) -> SO3:
        """Generate an `SO3` object from a particular angular
        parameterization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        """Construct an `AbstractPose` from an `SO3` object."""
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        offset_in_angstroms: Float[Array, "2"],
    ) -> Self:
        """Construct an `AbstractPose` from an `SO3` object and an
        in-plane translation vector.
        """
        return eqx.tree_at(
            lambda self: (
                self.offset_x_in_angstroms,
                self.offset_y_in_angstroms,
            ),
            cls.from_rotation(rotation),
            (
                offset_in_angstroms[..., 0],
                offset_in_angstroms[..., 1],
            ),
        )


class EulerAnglePose(AbstractPose, strict=True):
    r"""An `AbstractPose` represented by Euler angles.
    Angles are given in degrees, and the sequence of rotations is
    given by a zyz extrinsic rotations.
    """

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]

    view_phi: Float[Array, ""]
    view_theta: Float[Array, ""]
    view_psi: Float[Array, ""]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        view_phi: float | Float[Array, ""] = 0.0,
        view_theta: float | Float[Array, ""] = 0.0,
        view_psi: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `view_phi`: Angle to rotate about first rotation axis, which is the z axis.
        - `view_theta`: Angle to rotate about second rotation axis, which is the y axis.
        - `view_psi`: Angle to rotate about third rotation axis, which is the z axis.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.view_phi = jnp.asarray(view_phi)
        self.view_theta = jnp.asarray(view_theta)
        self.view_psi = jnp.asarray(view_psi)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate a `SO3` object from a set of Euler angles."""
        phi, theta, psi = self.view_phi, self.view_theta, self.view_psi
        # Convert to radians.
        phi = jnp.deg2rad(phi)
        theta = jnp.deg2rad(theta)
        psi = jnp.deg2rad(psi)
        # Get sequence of rotations.
        R1, R2, R3 = (
            SO3.from_z_radians(phi),
            SO3.from_y_radians(theta),
            SO3.from_z_radians(psi),
        )
        return R3 @ R2 @ R1

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        view_phi, view_theta, view_psi = convert_quaternion_to_euler_angles(
            rotation.wxyz,
            convention="zyz",
        )
        return cls(view_phi=view_phi, view_theta=view_theta, view_psi=view_psi)


class QuaternionPose(AbstractPose, strict=True):
    """An `AbstractPose` represented by unit quaternions."""

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]

    wxyz: Float[Array, "4"]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        wxyz: (
            tuple[float, float, float, float] | Float[np.ndarray, "4"] | Float[Array, "4"]
        ) = (1.0, 0.0, 0.0, 0.0),
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `wxyz`:
            The quaternion, represented as a vector $\\mathbf{q} = (q_w, q_x, q_y, q_z)$.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.wxyz = jnp.asarray(wxyz)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from the unit quaternion
        $\\mathbf{q} / |\\mathbf{q}|$.
        """
        # Generate SO3 object from unit quaternion
        R = SO3(wxyz=self.wxyz).normalize()
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        return cls(wxyz=rotation.wxyz)


class AxisAnglePose(AbstractPose, strict=True):
    """An `AbstractPose` parameterized in the axis-angle representation.

    The axis-angle representation parameterizes elements of the so3 algebra,
    which are skew-symmetric matrices, with the euler vector
    $\\boldsymbol{\\omega} = (\\omega_x, \\omega_y, \\omega_z)$.
    The magnitude of this vector is the angle, and the unit vector is the axis.

    In a `SO3` object, the euler vector is mapped to SO3 group elements using
    the matrix exponential.
    """

    offset_x_in_angstroms: Float[Array, ""]
    offset_y_in_angstroms: Float[Array, ""]

    euler_vector: Float[Array, "3"]

    def __init__(
        self,
        offset_x_in_angstroms: float | Float[Array, ""] = 0.0,
        offset_y_in_angstroms: float | Float[Array, ""] = 0.0,
        euler_vector: (
            tuple[float, float, float] | Float[np.ndarray, "3"] | Float[Array, "3"]
        ) = (0.0, 0.0, 0.0),
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `euler_vector`:
            The axis-angle parameterization, represented with the euler
            vector $\\boldsymbol{\\omega}$.
        """
        self.offset_x_in_angstroms = jnp.asarray(offset_x_in_angstroms)
        self.offset_y_in_angstroms = jnp.asarray(offset_y_in_angstroms)
        self.euler_vector = jnp.asarray(euler_vector)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from an euler vector using the exponential map."""
        # Convert degrees to radians
        euler_vector = jnp.deg2rad(self.euler_vector)
        # Project the tangent vector onto the manifold with
        # the exponential map
        R = SO3.exp(euler_vector)
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        # Compute the euler vector from the logarithmic map
        euler_vector = jnp.rad2deg(rotation.log())
        return cls(euler_vector=euler_vector)

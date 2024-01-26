import pytest
import numpy as np
from jax import config

from cryojax.simulator import make_euler_rotation

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "phi, theta, psi",
    [(10, 90, 170)],
    # [(10, 80, -20), (1.2, -90.5, 67), (-50, 62, -21)],
)
def test_euler_matrix(phi, theta, psi):
    """Test zyz rotation matrix"""
    # Hard code zyz rotation matrix from cisTEM convention
    phi, theta, psi = [np.deg2rad(angle) for angle in [phi, theta, psi]]
    matrix = np.zeros((3, 3))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    matrix[0, 0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi
    matrix[1, 0] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi
    matrix[2, 0] = -sin_theta * cos_psi
    matrix[0, 1] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi
    matrix[1, 1] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi
    matrix[2, 1] = sin_theta * sin_psi
    matrix[0, 2] = sin_theta * cos_phi
    matrix[1, 2] = sin_theta * sin_phi
    matrix[2, 2] = cos_theta
    # Generate rotation
    rotation = make_euler_rotation(
        phi,
        theta,
        psi,
        convention="zyz",
        intrinsic=True,
        degrees=False,
    )
    np.testing.assert_allclose(rotation.as_matrix(), matrix, atol=1e-16)

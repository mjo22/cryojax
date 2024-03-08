import pytest

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import cryojax.simulator as cs
from jaxlie import SO3


def test_default_pose_arguments():
    euler = cs.EulerPose()
    quat = cs.QuaternionPose()
    matrix = cs.MatrixPose()
    axis_angle = cs.AxisAnglePose()
    exponential = cs.ExponentialPose()
    np.testing.assert_allclose(euler.rotation.as_matrix(), quat.rotation.as_matrix())
    np.testing.assert_allclose(euler.rotation.as_matrix(), matrix.rotation.as_matrix())
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), axis_angle.rotation.as_matrix()
    )
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), exponential.rotation.as_matrix()
    )


def test_translation_agreement():
    rotation = SO3(jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    offset = jnp.asarray((0.0, -1.4, 4.5))
    # euler = cs.EulerPose.from_rotation()
    quat = cs.QuaternionPose.from_rotation_and_translation(rotation, offset)
    matrix = cs.MatrixPose.from_rotation_and_translation(rotation, offset)
    exponential = cs.ExponentialPose.from_rotation_and_translation(rotation, offset)
    np.testing.assert_allclose(quat.rotation.as_matrix(), matrix.rotation.as_matrix())
    np.testing.assert_allclose(
        quat.rotation.as_matrix(), exponential.rotation.as_matrix()
    )
    np.testing.assert_allclose(quat.offset, matrix.offset)
    np.testing.assert_allclose(quat.offset, exponential.offset)


def test_pose_conversion():
    wxyz = jnp.asarray((1.0, 2.0, 3.0, 0.0))
    rotation = SO3(wxyz).normalize()
    quat = cs.QuaternionPose.from_rotation(rotation)
    matrix = cs.MatrixPose.from_rotation(rotation)
    euler = cs.EulerPose.from_rotation(rotation)
    # exponential = cs.ExponentialPose.from_rotation(rotation)
    # axis_angle = cs.AxisAnglePose.from_rotation(rotation)
    np.testing.assert_allclose(quat.rotation.as_matrix(), matrix.rotation.as_matrix())
    # np.testing.assert_allclose(
    #   quat.rotation.as_matrix(), exponential.rotation.normalize().as_matrix()
    # )
    np.testing.assert_allclose(quat.rotation.as_matrix(), euler.rotation.as_matrix())
    # np.testing.assert_allclose(quat.rotation.as_matrix(), axis_angle.rotation.as_matrix())


def test_default_pose_images(noiseless_model):
    euler = cs.EulerPose()
    quat = cs.QuaternionPose()

    model_euler = eqx.tree_at(lambda m: m.specimen.pose, noiseless_model, euler)
    model_quat = eqx.tree_at(lambda m: m.specimen.pose, noiseless_model, quat)
    np.testing.assert_allclose(model_euler.render(), model_quat.render())


def test_axis_angle_euler_agreement():
    angle = 2.0
    euler_x = cs.EulerPose(view_phi=angle, convention="xyz")
    euler_y = cs.EulerPose(view_theta=angle, convention="xyz")
    euler_z = cs.EulerPose(view_psi=angle, convention="xyz")
    aa_x = cs.AxisAnglePose(axis=(1.0, 0.0, 0.0), angle=angle)
    aa_y = cs.AxisAnglePose(axis=(0.0, 1.0, 0.0), angle=angle)
    aa_z = cs.AxisAnglePose(axis=(0.0, 0.0, 1.0), angle=angle)
    np.testing.assert_allclose(euler_x.rotation.as_matrix(), aa_x.rotation.as_matrix())
    np.testing.assert_allclose(euler_y.rotation.as_matrix(), aa_y.rotation.as_matrix())
    np.testing.assert_allclose(euler_z.rotation.as_matrix(), aa_z.rotation.as_matrix())


@pytest.mark.parametrize("convention", ["zyz", "zyx", "zxz", "xyz"])
def test_euler_angle_conversion(convention):
    phi, theta, psi = 2.0, -15.0, 40.0
    pose = cs.EulerPose(
        view_phi=phi, view_theta=theta, view_psi=psi, convention=convention
    )
    converted_pose = cs.EulerPose.from_rotation(pose.rotation, convention=convention)
    np.testing.assert_allclose(
        np.asarray((phi, theta, psi)),
        np.asarray(
            (
                converted_pose.view_phi,
                converted_pose.view_theta,
                converted_pose.view_psi,
            )
        ),
    )

import pytest

import equinox as eqx
import numpy as np
import cryojax.simulator as cs


def test_default_pose_arguments():
    euler = cs.EulerPose()
    quat = cs.QuaternionPose()
    matrix = cs.MatrixPose()
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), quat.rotation.as_matrix()
    )
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), matrix.rotation.as_matrix()
    )


def test_default_pose_images(noiseless_model):
    euler = cs.EulerPose()
    quat = cs.QuaternionPose()
    model_euler = eqx.tree_at(
        lambda m: m.ensemble.pose, noiseless_model, euler
    )
    model_quat = eqx.tree_at(lambda m: m.ensemble.pose, noiseless_model, quat)
    np.testing.assert_allclose(model_euler.render(), model_quat.render())

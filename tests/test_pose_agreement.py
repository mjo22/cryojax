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
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), quat.rotation.as_matrix()
    )
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), matrix.rotation.as_matrix()
    )


def test_from_jaxlie():
    rotation = SO3(jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    offset = jnp.asarray((0.0, -1.4, 4.5))
    # euler = cs.EulerPose.from_rotation()
    quat = cs.QuaternionPose.from_rotation_and_translation(rotation, offset)
    matrix = cs.MatrixPose.from_rotation_and_translation(rotation, offset)
    np.testing.assert_allclose(
        quat.rotation.as_matrix(), matrix.rotation.as_matrix()
    )
    np.testing.assert_allclose(quat.offset, matrix.offset)


def test_default_pose_images(noiseless_model):
    euler = cs.EulerPose()
    quat = cs.QuaternionPose()
    model_euler = eqx.tree_at(
        lambda m: m.specimen.pose, noiseless_model, euler
    )
    model_quat = eqx.tree_at(lambda m: m.specimen.pose, noiseless_model, quat)
    np.testing.assert_allclose(model_euler.render(), model_quat.render())

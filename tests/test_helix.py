import pytest

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cryojax.simulator as cs


def build_helix(sample_subunit_mrc_path) -> cs.Helix:
    subunit_density = cs.FourierVoxelGrid.from_file(sample_subunit_mrc_path)
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    pose = cs.EulerPose(*r_0)
    ensemble = cs.Ensemble(subunit_density, pose)
    return cs.Helix(
        ensemble, rise=21.8, twist=29.4, n_start=6, n_subunits_per_start=1
    )


def build_helix_with_conformation(sample_subunit_mrc_path) -> cs.Helix:
    subunit_density = cs.FourierVoxelGrid.from_list(
        [
            cs.FourierVoxelGrid.from_file(sample_subunit_mrc_path)
            for _ in range(2)
        ]
    )
    n_start, n_subunits_per_start = 6, 1
    pose = cs.EulerPose()
    ensemble = cs.Ensemble(subunit_density, pose)
    conformation = cs.Conformation(
        np.random.choice(2, n_start * n_subunits_per_start)
    )
    print(conformation)
    return cs.Helix(
        ensemble,
        conformation=conformation,
        rise=21.8,
        twist=29.4,
        n_start=n_start,
        n_subunits_per_start=n_subunits_per_start,
    )


def test_superposition_pipeline_without_conformation(
    sample_subunit_mrc_path, scattering
):
    helix = build_helix(sample_subunit_mrc_path)
    pipeline = cs.SuperpositionPipeline(
        scattering=scattering, ensemble=helix.subunits
    )
    image = pipeline.render()


def test_superposition_pipeline_with_conformation(
    sample_subunit_mrc_path, scattering
):
    helix = build_helix_with_conformation(sample_subunit_mrc_path)
    pipeline = cs.SuperpositionPipeline(
        scattering=scattering, ensemble=helix.subunits
    )
    image = pipeline.render()


@pytest.mark.parametrize(
    "rotation_angle",
    [360.0 / 6, 2 * 360.0 / 6],
)
def test_c6_rotation(sample_subunit_mrc_path, scattering, rotation_angle):
    helix = build_helix(sample_subunit_mrc_path)

    @jax.jit
    def compute_rotated_image(helix, scattering, angles):
        where = lambda m: (m.pose.view_phi, m.pose.view_theta, m.pose.view_psi)
        helix = eqx.tree_at(where, helix, angles)
        pipeline = cs.SuperpositionPipeline(
            scattering=scattering, ensemble=helix.subunits
        )
        return pipeline.render()

    null_angles, rotated_angles = jnp.asarray((0.0, 0.0, 0.0)), jnp.asarray(
        (rotation_angle, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        compute_rotated_image(helix, scattering, null_angles),
        compute_rotated_image(helix, scattering, rotated_angles),
    )

import pytest

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cryojax.simulator as cs
from cryojax.io import read_volume_with_voxel_size_from_mrc
from jax import config

config.update("jax_enable_x64", True)


def build_helix(sample_subunit_mrc_path, n_subunits_per_start) -> cs.Helix:
    real_voxel_grid, voxel_size = read_volume_with_voxel_size_from_mrc(
        sample_subunit_mrc_path
    )
    subunit_density = cs.FourierVoxelGrid.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=2
    )
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerPose(*r_0)
    subunit = cs.Specimen(subunit_density, subunit_pose)
    return cs.Helix(
        subunit,
        rise=21.8,
        twist=29.4,
        n_start=6,
        n_subunits=n_subunits_per_start * 6,
    )


def build_helix_with_conformation(
    sample_subunit_mrc_path, n_subunits_per_start
) -> cs.Helix:
    subunit_density = tuple(
        [
            cs.FourierVoxelGrid.from_real_voxel_grid(
                *read_volume_with_voxel_size_from_mrc(sample_subunit_mrc_path)
            )
            for _ in range(2)
        ]
    )
    n_start = 6
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerPose(*r_0)
    subunit = cs.DiscreteEnsemble(
        subunit_density, subunit_pose, conformation=cs.DiscreteConformation(0)
    )
    conformation = cs.DiscreteConformation(
        np.random.choice(2, n_start * n_subunits_per_start)
    )
    return cs.Helix(
        subunit,
        conformation=conformation,
        rise=21.8,
        twist=29.4,
        n_start=n_start,
        n_subunits=n_subunits_per_start * 6,
    )


def test_superposition_pipeline_without_conformation(
    sample_subunit_mrc_path, integrator
):
    helix = build_helix(sample_subunit_mrc_path, 1)
    pipeline = cs.AssemblyPipeline(integrator=integrator, assembly=helix)
    image = pipeline.render()
    stochastic_image = pipeline.sample(jax.random.PRNGKey(0))


def test_superposition_pipeline_with_conformation(sample_subunit_mrc_path, integrator):
    helix = build_helix_with_conformation(sample_subunit_mrc_path, 2)
    pipeline = cs.AssemblyPipeline(integrator=integrator, assembly=helix)
    image = pipeline.render()
    stochastic_image = pipeline.sample(jax.random.PRNGKey(0))


@pytest.mark.parametrize(
    "rotation_angle, n_subunits_per_start",
    [(360.0 / 6, 1), (2 * 360.0 / 6, 1), (360.0 / 6, 2)],
)
def test_c6_rotation(
    sample_subunit_mrc_path, integrator, rotation_angle, n_subunits_per_start
):
    helix = build_helix(sample_subunit_mrc_path, n_subunits_per_start)

    @jax.jit
    def compute_rotated_image(helix, scattering, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(integrator=scattering, assembly=helix)
        return pipeline.render(normalize=True)

    np.testing.assert_allclose(
        compute_rotated_image(helix, integrator, cs.EulerPose()),
        compute_rotated_image(helix, integrator, cs.EulerPose(view_phi=rotation_angle)),
    )


@pytest.mark.parametrize(
    "translation, euler_angles",
    [
        ((0.0, 0.0), (60.0, 100.0, -40.0)),
        ((1.0, -3.0), (10.0, 50.0, 100.0)),
    ],
)
def test_agree_with_3j9g_assembly(
    sample_subunit_mrc_path, potential, integrator, translation, euler_angles
):
    helix = build_helix(sample_subunit_mrc_path, 2)
    specimen_39jg = cs.Specimen(potential)

    @jax.jit
    def compute_rotated_image_with_helix(helix, scattering, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(integrator=scattering, assembly=helix)
        return pipeline.render(normalize=True)

    @jax.jit
    def compute_rotated_image_with_3j9g(specimen, scattering, pose):
        specimen = eqx.tree_at(lambda m: m.pose, specimen, pose)
        pipeline = cs.ImagePipeline(integrator=scattering, specimen=specimen)
        return pipeline.render(normalize=True)

    pose = cs.EulerPose(*translation, 0.0, *euler_angles)
    reference_image = compute_rotated_image_with_3j9g(
        specimen_39jg, integrator, cs.EulerPose()
    )
    assembled_image = compute_rotated_image_with_helix(helix, integrator, pose)
    test_image = compute_rotated_image_with_3j9g(specimen_39jg, integrator, pose)
    assert np.std(assembled_image - test_image) < 10 * np.std(
        assembled_image - reference_image
    )


def test_transform_by_rise_and_twist(sample_subunit_mrc_path, pixel_size):
    helix = build_helix(sample_subunit_mrc_path, 12)
    scattering = cs.FourierSliceExtract(
        cs.ImageConfig((50, 20), pixel_size, pad_scale=6)
    )

    @jax.jit
    def compute_rotated_image(helix, scattering, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(integrator=scattering, assembly=helix)
        return pipeline.render(normalize=True)

    np.testing.assert_allclose(
        compute_rotated_image(
            helix,
            scattering,
            cs.EulerPose(view_phi=0.0, view_theta=90.0, view_psi=0.0),
        ),
        compute_rotated_image(
            helix,
            scattering,
            cs.EulerPose(
                view_phi=helix.twist,
                view_theta=90.0,
                view_psi=0.0,
                offset_x=helix.rise,
            ),
        ),
        atol=1e-1,
    )

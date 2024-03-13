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
    subunit_density = cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=2
    )
    integrator = cs.FourierSliceExtract()
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerAnglePose(*r_0)
    subunit = cs.Specimen(subunit_density, integrator, subunit_pose)
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
            cs.FourierVoxelGridPotential.from_real_voxel_grid(
                *read_volume_with_voxel_size_from_mrc(sample_subunit_mrc_path)
            )
            for _ in range(2)
        ]
    )
    n_start = 6
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerAnglePose(*r_0)
    integrator = cs.FourierSliceExtract()
    subunit = cs.DiscreteEnsemble(
        subunit_density,
        integrator,
        subunit_pose,
        conformation=cs.DiscreteConformation(0),
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


def test_superposition_pipeline_without_conformation(sample_subunit_mrc_path, config):
    helix = build_helix(sample_subunit_mrc_path, 1)
    pipeline = cs.AssemblyPipeline(config=config, assembly=helix)
    _ = pipeline.render()
    a = pipeline.sample(jax.random.PRNGKey(0))


def test_superposition_pipeline_with_conformation(sample_subunit_mrc_path, config):
    helix = build_helix_with_conformation(sample_subunit_mrc_path, 2)
    pipeline = cs.AssemblyPipeline(config=config, assembly=helix)
    _ = pipeline.render()
    _ = pipeline.sample(jax.random.PRNGKey(0))


"""
@pytest.mark.parametrize(
    "rotation_angle, n_subunits_per_start",
    [(360.0 / 6, 1), (2 * 360.0 / 6, 1), (360.0 / 6, 2)],
)
def test_c6_rotation(sample_subunit_mrc_path, rotation_angle, n_subunits_per_start):
    helix = build_helix(sample_subunit_mrc_path, n_subunits_per_start)

    @jax.jit
    def compute_rotated_image(config, helix, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(config=config, assembly=helix)
        return pipeline.render(normalize=True)

    np.testing.assert_allclose(
        compute_rotated_image(config, helix, cs.EulerAnglePose()),
        compute_rotated_image(
            config, helix, cs.EulerAnglePose(view_phi=rotation_angle)
        ),
    )
"""


@pytest.mark.parametrize(
    "translation, euler_angles",
    [
        ((0.0, 0.0), (60.0, 100.0, -40.0)),
        ((1.0, -3.0), (10.0, 50.0, 100.0)),
    ],
)
def test_agree_with_3j9g_assembly(
    sample_subunit_mrc_path, potential, config, translation, euler_angles
):
    helix = build_helix(sample_subunit_mrc_path, 2)
    specimen_39jg = cs.Specimen(potential, helix.subunit.integrator)

    @jax.jit
    def compute_rotated_image_with_helix(helix, config, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(config=config, assembly=helix)
        return pipeline.render(normalize=True)

    @jax.jit
    def compute_rotated_image_with_3j9g(specimen, config, pose):
        specimen = eqx.tree_at(lambda m: m.pose, specimen, pose)
        pipeline = cs.ImagePipeline(config=config, specimen=specimen)
        return pipeline.render(normalize=True)

    pose = cs.EulerAnglePose(*translation, 0.0, *euler_angles)
    reference_image = compute_rotated_image_with_3j9g(
        specimen_39jg, config, cs.EulerAnglePose()
    )
    assembled_image = compute_rotated_image_with_helix(helix, config, pose)
    test_image = compute_rotated_image_with_3j9g(specimen_39jg, config, pose)
    assert np.std(assembled_image - test_image) < 10 * np.std(
        assembled_image - reference_image
    )


def test_transform_by_rise_and_twist(sample_subunit_mrc_path, pixel_size):
    helix = build_helix(sample_subunit_mrc_path, 12)
    config = cs.ImageConfig((50, 20), pixel_size, pad_scale=6)

    @jax.jit
    def compute_rotated_image(config, helix, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        pipeline = cs.AssemblyPipeline(config=config, assembly=helix)
        return pipeline.render(normalize=True)

    np.testing.assert_allclose(
        compute_rotated_image(
            config,
            helix,
            cs.EulerAnglePose(view_phi=0.0, view_theta=90.0, view_psi=0.0),
        ),
        compute_rotated_image(
            config,
            helix,
            cs.EulerAnglePose(
                view_phi=helix.twist,
                view_theta=90.0,
                view_psi=0.0,
                offset_x_in_angstroms=helix.rise,
            ),
        ),
        atol=1e-1,
    )

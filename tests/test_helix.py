import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cryojax.simulator as cs
from cryojax.image import irfftn, normalize_image
from cryojax.io import read_array_with_spacing_from_mrc


def build_helix(sample_subunit_mrc_path, n_subunits_per_start) -> cs.HelicalAssembly:
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(
        sample_subunit_mrc_path
    )
    subunit_density = cs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=2
    )
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerAnglePose(*r_0)
    subunit = cs.SingleStructureEnsemble(subunit_density, subunit_pose)
    return cs.HelicalAssembly(
        subunit,
        rise=21.8,
        twist=29.4,
        n_start=6,
        n_subunits=n_subunits_per_start * 6,
    )


def build_helix_with_conformation(
    sample_subunit_mrc_path, n_subunits_per_start
) -> cs.HelicalAssembly:
    subunit_density = tuple(
        [
            cs.FourierVoxelGridPotential.from_real_voxel_grid(
                *read_array_with_spacing_from_mrc(sample_subunit_mrc_path)
            )
            for _ in range(2)
        ]
    )
    n_start = 6
    r_0 = jnp.asarray([-88.70895129, 9.75357114, 0.0], dtype=float)
    subunit_pose = cs.EulerAnglePose(*r_0)
    subunit = cs.DiscreteStructuralEnsemble(
        subunit_density,
        subunit_pose,
        conformation=cs.DiscreteConformationalVariable(0),
    )
    conformation = jax.vmap(lambda value: cs.DiscreteConformationalVariable(value))(
        np.random.choice(2, n_start * n_subunits_per_start)
    )
    return cs.HelicalAssembly(
        subunit,
        conformation=conformation,
        rise=21.8,
        twist=29.4,
        n_start=n_start,
        n_subunits=n_subunits_per_start * 6,
    )


def test_superposition_pipeline_without_conformation(sample_subunit_mrc_path, config):
    helix = build_helix(sample_subunit_mrc_path, 1)
    projection_method = cs.FourierSliceExtraction()
    transfer_theory = cs.ContrastTransferTheory(cs.IdealContrastTransferFunction())
    theory = cs.LinearSuperpositionScatteringTheory(
        helix, projection_method, transfer_theory
    )
    pipeline = cs.ContrastImagingPipeline(
        instrument_config=config, scattering_theory=theory
    )
    _ = pipeline.render()
    _ = pipeline.render(jax.random.PRNGKey(0))


def test_superposition_pipeline_with_conformation(sample_subunit_mrc_path, config):
    helix = build_helix_with_conformation(sample_subunit_mrc_path, 2)
    projection_method = cs.FourierSliceExtraction()
    transfer_theory = cs.ContrastTransferTheory(cs.IdealContrastTransferFunction())
    theory = cs.LinearSuperpositionScatteringTheory(
        helix, projection_method, transfer_theory
    )
    pipeline = cs.ContrastImagingPipeline(
        instrument_config=config, scattering_theory=theory
    )
    _ = pipeline.render()
    _ = pipeline.render(jax.random.PRNGKey(0))


@pytest.mark.parametrize(
    "rotation_angle, n_subunits_per_start",
    [(360.0 / 6, 1), (2 * 360.0 / 6, 1), (360.0 / 6, 2)],
)
def test_c6_rotation(
    sample_subunit_mrc_path, config, rotation_angle, n_subunits_per_start
):
    helix = build_helix(sample_subunit_mrc_path, n_subunits_per_start)
    projection_method = cs.FourierSliceExtraction()
    transfer_theory = cs.ContrastTransferTheory(cs.IdealContrastTransferFunction())
    theory = cs.LinearSuperpositionScatteringTheory(
        helix, projection_method, transfer_theory
    )
    pipeline = cs.ContrastImagingPipeline(
        instrument_config=config, scattering_theory=theory
    )

    @eqx.filter_jit
    def compute_rotated_image(pipeline, pose):
        pipeline = eqx.tree_at(
            lambda m: m.scattering_theory.structural_ensemble_batcher.pose,
            pipeline,
            pose,
        )
        return normalize_image(pipeline.render())

    np.testing.assert_allclose(
        compute_rotated_image(pipeline, cs.EulerAnglePose()),
        compute_rotated_image(pipeline, cs.EulerAnglePose(view_phi=rotation_angle)),
    )


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
    specimen_39jg = cs.SingleStructureEnsemble(potential, cs.EulerAnglePose())
    superposition_theory = cs.LinearSuperpositionScatteringTheory(
        helix,
        cs.FourierSliceExtraction(),
        cs.ContrastTransferTheory(cs.IdealContrastTransferFunction()),
    )
    theory = cs.WeakPhaseScatteringTheory(
        specimen_39jg,
        cs.FourierSliceExtraction(),
        cs.ContrastTransferTheory(cs.IdealContrastTransferFunction()),
    )
    pipeline_for_assembly = cs.ContrastImagingPipeline(
        instrument_config=config, scattering_theory=superposition_theory
    )
    pipeline_for_3j9g = cs.ContrastImagingPipeline(
        instrument_config=config, scattering_theory=theory
    )

    @eqx.filter_jit
    def compute_rotated_image_with_helix(
        pipeline: cs.ContrastImagingPipeline, pose: cs.AbstractPose
    ):
        pipeline = eqx.tree_at(
            lambda m: m.scattering_theory.structural_ensemble_batcher.pose,
            pipeline,
            pose,
        )
        return normalize_image(pipeline.render())

    @eqx.filter_jit
    def compute_rotated_image_with_3j9g(
        pipeline: cs.ContrastImagingPipeline, pose: cs.AbstractPose
    ):
        pipeline = eqx.tree_at(
            lambda m: m.scattering_theory.structural_ensemble.pose, pipeline, pose
        )
        return normalize_image(pipeline.render())

    pose = cs.EulerAnglePose(*translation, 0.0, *euler_angles)
    reference_image = compute_rotated_image_with_3j9g(
        pipeline_for_3j9g, cs.EulerAnglePose()
    )
    assembled_image = compute_rotated_image_with_helix(pipeline_for_assembly, pose)
    test_image = compute_rotated_image_with_3j9g(pipeline_for_3j9g, pose)
    assert np.std(assembled_image - test_image) < 10 * np.std(
        assembled_image - reference_image
    )


def test_transform_by_rise_and_twist(sample_subunit_mrc_path, pixel_size):
    helix = build_helix(sample_subunit_mrc_path, 12)
    config = cs.InstrumentConfig((50, 20), pixel_size, 300.0, pad_scale=6)

    @eqx.filter_jit
    def compute_rotated_image(config, helix, pose):
        helix = eqx.tree_at(lambda m: m.pose, helix, pose)
        theory = cs.LinearSuperpositionScatteringTheory(
            helix,
            cs.FourierSliceExtraction(),
            cs.ContrastTransferTheory(cs.IdealContrastTransferFunction()),
        )
        return config.crop_to_shape(
            irfftn(
                theory.compute_fourier_phase_shifts_at_exit_plane(config),
                s=config.padded_shape,
            )
        )  # noqa: E501

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

import pytest
import jax.numpy as jnp
import numpy as np
from jax import config
from pycistem.core import CTF as cisCTF, Image, AnglesAndShifts

import cryojax.simulator as cs
from cryojax.io import read_array_with_spacing_from_mrc
from cryojax.simulator import CTF, make_euler_rotation
from cryojax.image import powerspectrum, irfftn
from cryojax.coordinates import make_frequencies, cartesian_to_polar

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "defocus1,defocus2,asti_angle,kV,cs,ac,pixel_size",
    [
        (12000, 12000, 0.0, 300.0, 2.7, 0.07, 1.0),
        (12000, 12000, 0.0, 200.0, 0.01, 0.12, 1.3),
        (1200, 1200, 0.0, 300.0, 2.7, 0.07, 1.5),
        (24000, 12000, 30.0, 300.0, 2.7, 0.07, 0.9),
        (24000, 24000, 0.0, 300.0, 2.7, 0.07, 2.0),
        (9000, 7000, 180.0, 300.0, 2.7, 0.07, 1.0),
        (12000, 9000, 0.0, 200.0, 2.7, 0.07, 0.9),
        (12000, 12000, 60.0, 200.0, 2.7, 0.02, 0.75),
        (12000, 3895, 45.0, 200.0, 2.7, 0.07, 2.2),
    ],
)
def test_ctf_with_cistem(defocus1, defocus2, asti_angle, kV, cs, ac, pixel_size):
    """Test CTF model against cisTEM.

    Modified from https://github.com/jojoelfe/contrasttransferfunction"""
    shape = (512, 512)
    freqs = make_frequencies(shape, pixel_size)
    k_sqr, theta = cartesian_to_polar(freqs, square=True)
    # Compute cryojax CTF
    optics = CTF(
        defocus_u_in_angstroms=defocus1,
        defocus_v_in_angstroms=defocus2,
        astigmatism_angle=asti_angle,
        voltage_in_kilovolts=kV,
        spherical_aberration_in_mm=cs,
        amplitude_contrast_ratio=ac,
    )
    ctf = np.array(optics(freqs))
    # Compute cisTEM CTF
    cisTEM_optics = cisCTF(
        kV=kV,
        cs=cs,
        ac=ac,
        defocus1=defocus1,
        defocus2=defocus2,
        astig_angle=asti_angle,
        pixel_size=pixel_size,
    )
    cisTEM_ctf = np.vectorize(
        lambda k_sqr, theta: cisTEM_optics.Evaluate(k_sqr, theta)
    )(k_sqr.ravel() * pixel_size**2, theta.ravel()).reshape(freqs.shape[0:2])
    cisTEM_ctf[0, 0] = 0.0

    # Compute cryojax and cisTEM power spectrum
    radial_freqs = jnp.linalg.norm(freqs, axis=-1)
    spectrum1D, _ = powerspectrum(
        ctf, radial_freqs, pixel_size, k_max=1 / (2 * pixel_size)
    )
    cisTEM_spectrum1D, _ = powerspectrum(
        cisTEM_ctf, radial_freqs, pixel_size, k_max=1 / (2 * pixel_size)
    )

    np.testing.assert_allclose(ctf, cisTEM_ctf, atol=5e-2)
    np.testing.assert_allclose(spectrum1D, cisTEM_spectrum1D, atol=5e-3)


@pytest.mark.parametrize(
    "phi, theta, psi",
    [(10, 90, 170)],
    # [(10, 80, -20), (1.2, -90.5, 67), (-50, 62, -21)],
)
def test_euler_matrix_with_cistem(phi, theta, psi):
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
    # Generate rotation that matches this rotation matrix
    rotation = make_euler_rotation(phi, theta, psi, convention="zyz", degrees=False)
    np.testing.assert_allclose(rotation.as_matrix(), matrix.T, atol=1e-12)


@pytest.mark.parametrize(
    "phi, theta, psi",
    [(10, 90, 170)],
)
def test_compute_projection_with_cistem(phi, theta, psi, sample_mrc_path, pixel_size):
    # cryojax
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    potential = cs.FourierVoxelGrid.from_real_voxel_grid(real_voxel_grid, voxel_size)
    pose = cs.EulerPose(view_phi=phi, view_theta=theta, view_psi=psi)
    specimen = cs.Specimen(potential, pose)
    box_size = potential.shape[0]
    config = cs.ImageConfig((box_size, box_size), pixel_size)
    scattering = cs.FourierSliceExtract(config)
    pipeline = cs.ImagePipeline(specimen, scattering)
    cryojax_projection = irfftn(
        pipeline.render(get_real=False).at[0, 0].set(0.0 + 0.0j)
        / np.sqrt(np.prod(config.shape)),
        s=config.padded_shape,
    )
    # pycistem
    pycistem_volume = _load_pycistem_template(sample_mrc_path, box_size)
    pycistem_angles = AnglesAndShifts()
    pycistem_angles.Init(phi, theta, psi, 0.0, 0.0)
    pycistem_model = _compute_projection(pycistem_volume, pycistem_angles, box_size)
    pycistem_projection = np.asarray(pycistem_model.real_values)

    np.testing.assert_allclose(cryojax_projection, pycistem_projection, atol=1e-5)


def _load_pycistem_template(filename, box_size):
    """Load pycistem template in fourier space."""
    volume = Image()
    volume.QuickAndDirtyReadSlices(filename, 1, box_size)
    volume.ForwardFFT(True)
    volume.ZeroCentralPixel()
    volume.SwapRealSpaceQuadrants()

    return volume


def _compute_projection(volume, angles, box_size):
    """Compute scattering projection of pycistem volume."""
    projection = Image()
    projection.Allocate(box_size, box_size, False)

    volume.ExtractSlice(projection, angles, 1.0, False)
    projection.PhaseShift(angles.ReturnShiftX(), angles.ReturnShiftY(), 0.0)

    projection.SwapRealSpaceQuadrants()
    projection.BackwardFFT()

    return projection

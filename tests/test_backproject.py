import pytest
import jax
import jax.numpy as jnp
import numpy as np
from cryojax.reconstruct import backproject
import cryojax.simulator as cs
from cryojax.utils import make_frequencies, cartesian_to_polar, fftn, ifftn


@pytest.mark.parametrize(
    "defocus1,defocus2,asti_angle,kV,cs,ac,pixel_size",
    [
        (12000, 12000, 0.0, 300.0, 2.7, 0.07, 1.0),
        (12000, 12000, 0.0, 200.0, 0.01, 0.12, 1.3),
    ],
)
def test_wiener_filter_divides_by_ctf(
    defocus1, defocus2, asti_angle, kV, cs, ac, pixel_size
):
    N = 512
    shape = (N, N)
    freqs = make_frequencies(shape, pixel_size, half_space=False)
    optics = cs.CTFOptics(
        defocus_u=defocus1,
        defocus_v=defocus2,
        defocus_angle=asti_angle,
        voltage=kV,
        spherical_aberration=cs,
        amplitude_contrast=ac,
    )
    ctf = np.array(optics(freqs))

    noise_level = 0
    wiener_filter = backproject.WeinerFilter(ctf, noise_level)

    image = jnp.arange(N * N).reshape(shape) + 1
    image_f = jnp.fft.fftshift(fftn(image))

    image_ctf_f = image_f * ctf

    image_deconvctf_f = wiener_filter(image_ctf_f)
    image_deconvctf_r = ifftn(jnp.fft.ifftshift(image_deconvctf_f)).real

    image_deconvctf_bydivide_f = image_ctf_f / ctf
    image_deconvctf_bydivide_r = ifftn(
        jnp.fft.ifftshift(image_deconvctf_bydivide_f)
    ).real

    resid = image_deconvctf_bydivide_r - image_deconvctf_r
    test_close_every_pixel = resid / image
    np.testing.assert_allclose(test_close_every_pixel, 0, atol=1e-3)


def test_filtered_backprojection():
    # demo filtered backprojection
    from scipy.spatial.transform import Rotation

    n_slices = 1000
    filename = "hackathon/emd-3683.mrc"
    random_rotations = Rotation.random(n_slices).as_euler(
        "zyz", degrees=True
    )  # np.random.uniform(-180,180, size=(n_slices,3))
    random_df = np.random.uniform(800, 1000, size=n_slices)

    density = cs.VoxelGrid.from_file(
        filename, config=dict(pad_scale=1, crop_scale=1)
    )
    shape = density.weights.shape[:2]
    pad_scale = 1
    pixel_size = 1

    n_pix_render = shape[0]
    manager = cs.ImageManager(shape=shape, pad_scale=pad_scale)
    scattering = cs.FourierSliceExtract(manager, pixel_size=pixel_size)
    exposure = cs.UniformExposure(N=1.0, mu=0.0)
    detector = cs.GaussianDetector(variance=cs.Constant(30.0))

    freq_xy = jnp.fft.fftshift(
        make_frequencies((n_pix_render, n_pix_render), half_space=False)
    )
    slices_f = jnp.zeros(
        (n_slices, n_pix_render, n_pix_render), dtype=jnp.complex64
    )
    poses = []

    for idx in range(n_slices):
        pose = cs.EulerPose(
            view_phi=random_rotations[idx, 0],
            view_theta=random_rotations[idx, 1],
            view_psi=random_rotations[idx, 2],
        )
        poses.append(pose)
        ensemble = cs.Ensemble(density=density, pose=pose)
        df = random_df[idx]
        optics = cs.CTFOptics(
            defocus_u=df, defocus_v=df, amplitude_contrast=0.07
        )
        instrument_od = cs.Instrument(
            exposure=exposure, optics=optics, detector=detector
        )
        pipeline = cs.ImagePipeline(
            scattering=scattering, instrument=instrument_od, ensemble=ensemble
        )
        key = jax.random.PRNGKey(idx)
        image = pipeline.sample(key)
        image_f = jnp.fft.fftshift(fftn(image))

        ctf = optics(freq_xy)
        wiener_filter = backproject.WeinerFilter(ctf, 0.0001)
        image_deconv = wiener_filter(image_f)

        slices_f = slices_f.at[idx].set(image_deconv)

    fbp_vol_r = backproject.filtered_backprojection(
        deconolved_images_f=slices_f, poses=poses, to_real=True
    )
    fbp_vol_r -= fbp_vol_r.mean()
    fbp_vol_r /= fbp_vol_r.std()

    gt_vol_r = ifftn(density.weights).real
    gt_vol_r -= gt_vol_r.mean()
    gt_vol_r /= gt_vol_r.std()

    rel_resid = jnp.abs((gt_vol_r - fbp_vol_r)).mean() / gt_vol_r.std()
    print("residual_ratio", rel_resid)
    residual_ratio_tol = 0.9
    assert rel_resid < residual_ratio_tol


if __name__ == "__main__":
    test_filtered_backprojection()

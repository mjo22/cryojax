import pytest
import jax.numpy as jnp
import numpy as np
from cryojax.reconstruct import backproject
from cryojax.simulator import CTFOptics
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
    optics = CTFOptics(
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

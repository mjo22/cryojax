import pytest
import numpy as np
from pycistem.core import CTF

from cryojax.simulator import CTFOptics
from cryojax.utils import fftfreqs, cartesian_to_polar, powerspectrum


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
def test_ctf_with_cistem(
    defocus1, defocus2, asti_angle, kV, cs, ac, pixel_size
):
    """Test CTF model against cisTEM.

    Modified from https://github.com/jojoelfe/contrasttransferfunction"""
    shape = (512, 512)
    freqs = fftfreqs(shape, pixel_size=pixel_size)
    k_sqr, theta = cartesian_to_polar(freqs, square=True)
    # Compute cryojax CTF
    optics = CTFOptics(
        defocus_u=defocus1,
        defocus_v=defocus2,
        defocus_angle=asti_angle,
        voltage=kV,
        spherical_aberration=cs,
        amplitude_contrast=ac,
        envelope=None,
    )
    ctf = np.array(optics(freqs, normalize=False))
    # Compute cisTEM CTF
    cisTEM_optics = CTF(
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
    )(k_sqr.ravel() * pixel_size**2, theta.ravel()).reshape(shape)

    # Compute cryojax and cisTEM power spectrum
    spectrum1D, _ = powerspectrum(ctf, freqs, pixel_size=pixel_size)
    cisTEM_spectrum1D, _ = powerspectrum(
        cisTEM_ctf, freqs, pixel_size=pixel_size
    )

    np.testing.assert_allclose(ctf, cisTEM_ctf, atol=5e-2)
    np.testing.assert_allclose(spectrum1D, cisTEM_spectrum1D, atol=5e-3)

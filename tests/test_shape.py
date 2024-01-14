import pytest


@pytest.mark.parametrize("model", ["noisy_model", "filtered_and_masked_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model()
    padded_image = model(view_cropped=False)
    assert image.shape == model.scattering.manager.shape
    assert padded_image.shape == model.scattering.manager.padded_shape


@pytest.mark.parametrize("model", ["noisy_model", "filtered_and_masked_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model(get_real=False)
    padded_image = model(view_cropped=False, get_real=False)
    assert (
        image.shape == model.scattering.manager.frequency_grid.get().shape[0:2]
    )
    assert (
        padded_image.shape
        == model.scattering.manager.padded_frequency_grid.get().shape[0:2]
    )

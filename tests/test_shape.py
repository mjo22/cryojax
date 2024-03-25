import pytest


@pytest.mark.parametrize("model", ["noisy_model", "filtered_and_masked_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model.render()
    padded_image = model.render(view_cropped=False)
    assert image.shape == model.config.shape
    assert padded_image.shape == model.config.padded_shape


@pytest.mark.parametrize("model", ["noisy_model", "filtered_and_masked_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model.render(get_real=False)
    padded_image = model.render(view_cropped=False, get_real=False)
    assert image.shape == model.config.wrapped_frequency_grid_in_pixels.get().shape[0:2]
    assert (
        padded_image.shape
        == model.config.wrapped_padded_frequency_grid_in_pixels.get().shape[0:2]
    )

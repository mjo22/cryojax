import pytest


@pytest.mark.parametrize("model", ["noisy_model", "maskless_model"])
def test_shape(model, request):
    """Make sure shapes are as expected"""
    model = request.getfixturevalue(model)
    image = model()
    padded_image = model(view=False)
    assert image.shape == model.manager.shape
    assert padded_image.shape == model.manager.padded_shape

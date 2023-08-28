def test_shape(scattering_model):
    image = scattering_model()
    padded_image = scattering_model(view=False)
    assert image.shape == scattering_model.scattering.shape
    assert padded_image.shape == scattering_model.scattering.padded_shape

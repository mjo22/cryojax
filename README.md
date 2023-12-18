# Cryo-EM image simulation and analysis powered by JAX.

![Tests](https://github.com/mjo22/cryojax/actions/workflows/testing.yml/badge.svg)
![Lint](https://github.com/mjo22/cryojax/actions/workflows/black.yml/badge.svg)

This library is a modular framework for simulating forward models of cryo electron microscopy images. It is designed with 2D template matching analysis in mind, but it can be used generally. `cryojax` is, of course, built on [jax](https://github.com/google/jax). It also uses [equinox](https://github.com/patrick-kidger/equinox/) for modeling building, so `equinox` functionality is supported in `cryojax`.

## Summary

The core of this package is its ability to simulate cryo-EM images. Starting with a 3D electron density map, one can simulate a scattering process onto the imaging plane with modulation by the instrument optics. Images are then sampled from models of the noise or the corresponding log-likelihood is computed.

These models can be fed into standard sampling, optimization, and model building libraries in `jax`, such as [blackjax](https://github.com/blackjax-devs/blackjax), [optax](https://github.com/google-deepmind/optax), or [numpyro](https://github.com/pyro-ppl/numpyro). The `jax` ecosystem is rich and growing fast!

## Installation

Installing `cryojax` is simple. To start, I recommend creating a new virtual environment. For example, you could do this with `conda`.

```bash
conda create -n cryojax -c conda-forge python=3.10
```

Note that `python>=3.10` is required due to recent features in `dataclasses`. Now, [install JAX](https://github.com/google/jax#installation) with either CPU or GPU support.

Finally, install `cryojax`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/cryojax
cd cryojax
python -m pip install .
```

This will install the remaining dependencies, such as [equinox](https://github.com/patrick-kidger/equinox/) for jax-friendly dataclasses, [jaxlie](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, [mrcfile](https://github.com/ccpem/mrcfile) for I/O, and [dataclasses-json](https://github.com/lidatong/dataclasses-json) for serialization.

The [jax-finufft](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. These are included as an option for computing image projections. In this case, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Building a model

Please note that this library is currently experimental and the API is subject to change! The following is a basic workflow to generate an image with a gaussian white noise model.

First, instantiate the image formation method ("scattering") and its respective representation
of an electron density ("specimen").

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs

template = "example.mrc"
utils = cs.ImageManager(shape=(320, 320))
scattering = cs.FourierSliceScattering(utils)
density = cs.VoxelGrid.from_file(template)
```

Here, `template` is a 3D electron density map in MRC format. This could be taken from the [EMDB](https://www.ebi.ac.uk/emdb/), or rasterized from a [PDB](https://www.rcsb.org/). [cisTEM](https://github.com/timothygrant80/cisTEM) provides an excellent rasterization tool in its image simulation program. In the above example, a voxel electron density in fourier space is loaded and the fourier-slice projection theorem is initialized. We can now intstantiate the biological `Specimen`.

```python
pose = cs.EulerPose(view_phi=0.0, view_theta=0.0, view_psi=0.0)
specimen = cs.Specimen(density=density, pose=pose, resolution=1.1)
```

This is a container for the parameters and metadata stored in the electron density, the model for the `Pose`, and additional parameters such as the rasterization `resolution`.

Next, the model for the electron microscope. `Optics` and `Detector` models and their respective parameters are initialized. These are stored in the `Instrument` container.

```python
key = jax.random.PRNGKey(seed=0)
optics = cs.CTFOptics(defocus_u=10000.0, defocus_v=9800.0, defocus_angle=10.0)
detector = cs.GaussianDetector(key=key, pixel_size=1.1, variance=cs.Constant(1.0))
instrument = cs.Instrument(optics=optics, detector=detector)
```

Then, the `ImagePipeline` model is chosen. Here, we choose `GaussianImage`.

```python
model = cs.GaussianImage(scattering=scattering, specimen=specimen, instrument=instrument)
image = model.sample()
```

This computes an image using the noise model of the detector. One can also compute an image without the stochastic part of the model.

```python
image = model.render()
```

Imaging models also accept a series of `Filter`s and `Mask`s. For example, one could add a `LowpassFilter`, `WhiteningFilter`, and a `CircularMask`.

```python
micrograph = ...  # A micrograph used for whitening
filters = [cs.LowpassFilter(scattering.padded_shape, cutoff=1.0),  # Cutoff modes above Nyquist frequency
           cs.WhiteningFilter(scattering.padded_shape, micrograph=micrograph)]
masks = [cs.CircularMask(scattering.shape, radius=1.0)]           # Cutoff pixels above radius equal to (half) image size
model = cs.GaussianImage(scattering=scattering, specimen=specimen, instrument=instrument, filters=filters, masks=masks)
image = model.sample()
```

If a `GaussianImage` is passed `observed`, the model will instead compute the log likelihood.

```python
model = cs.GaussianImage(scattering=scattering, specimen=specimen, instrument=instrument)
log_likelihood = model.log_probability(observed)
```

Note that the user may need to do preprocessing of `observed`, such as applying the relevant `Filter`s and `Mask`s.

Additional components can be plugged into the image formation model. For example, modeling the solvent is supported through the `ImagePipeline`'s `Ice` model. Models for exposure to the electron beam are supported through the `Instrument`'s `Exposure` model.

For these more advanced examples, see the tutorials section of the repository. In general, `cryojax` is designed to be very extensible and new models can easily be implemented.

## Creating a loss function

In `jax`, we ultimately want to build a loss function and apply functional transformations to it. Assuming we have already globally configured our model components at our desired initial state, the below creates a loss function at an updated set of parameters. First, we must update the model.

```python

@jax.jit
def update_model(model, params):
    """
    Update the model with equinox.tree_at (https://docs.kidger.site/equinox/api/manipulation/#equinox.tree_at).
    """
    where = lambda model: (model.specimen.pose.view_phi, model.instrument.optics.defocus_u, model.instrument.detector.pixel_size)
    updated_model = eqx.tree_at(where, model, (params["view_phi"], params["defocus_u"], params["pixel_size"]))
    return updated_model
```

We can now create the loss and differentiate it with respect to the parameters.

```python
@jax.jit
@jax.value_and_grad
def loss(params, model, observed):
    model = update_model(model, params)
    return model.log_probability(observed)
```

Finally, we can evaluate an updated set of parameters.

```python
params = dict(view_phi=jnp.asarray(jnp.pi), defocus_u=jnp.asarray(9000.0), pixel_size=jnp.asarray(1.30))
log_likelihood, grad = loss(params, model, observed)
```

To summarize, this example creates a loss function at an updated set of `Pose`, `Optics`, and `Detector` parameters. In general, any `cryojax` `Module` may contain model parameters. One gotcha is just that the `ScatteringConfig`, `Filter`s, and `Mask`s all do computation upon initialization, so they should not be explicitly instantiated in the loss function evaluation. Another gotcha is that if the `model` is not passed as an argument to the loss, there may be long compilation times because the electron density will be treated as static. However, this may result in slight speedups.

In general, there are many ways to write loss functions. See the [equinox](https://github.com/patrick-kidger/equinox/) documentation for more use cases.

## Features

- Imaging models in `cryojax` support `jax` functional transformations, such as automatic differentiation with `grad`, paralellization with `vmap` and `pmap`, and just-in-time compilation with `jit`. Models also support GPU/TPU acceleration.
- `cryojax.Module`s, including `ImagePipeline` models, are JSON serializable thanks to the package `dataclasses-json`. The method `Module.dumps` serializes the object as a JSON string, and `Module.loads` instantiates it from the string. For example, write a model to disk with `model.dump("model.json")` and instantiate it with `cs.GaussianImage.load("model.json")`.
- A `cryojax.Module` is just an `equinox.Module` with added serialization functionality. Therefore, the entire `equinox` ecosystem is available for usage!

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

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

First, instantiate the `ScatteringModel` and its respective representation
of an `ElectronDensity`.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs

filename = "example.mrc"
density = cs.VoxelGrid.from_file(filename)
pixel_size = density.voxel_size
manager = cs.ImageManager(shape=(320, 320))
scattering = cs.FourierSliceExtract(manager, pixel_size=pixel_size)
```

Here, `filename` is a 3D electron density map in MRC format. This could be taken from the [EMDB](https://www.ebi.ac.uk/emdb/), or rasterized from a [PDB](https://www.rcsb.org/). [cisTEM](https://github.com/timothygrant80/cisTEM) provides an excellent rasterization tool in its image simulation program. In the above example, a voxel electron density in fourier space is loaded and the fourier-slice projection theorem is initialized. Note that we must explicitly set the pixel size of the projection image. Here, it is the same as the voxel size of the electron density. We can now instantiate the `Ensemble` of biological specimen.

```python
# Translations in Angstroms, angles in degrees
pose = cs.EulerPose(offset_x=5.0, offset_y=-3.0, view_phi=20.0, view_theta=80.0, view_psi=-10.0)
ensemble = cs.Ensemble(density=density, pose=pose)
```

Here, this holds the `ElectronDensity` and the model for the `Pose`. If instead a stack of `ElectronDensity` is loaded, the `Ensemble` can be evaluated at a particular conformation.

```python
filenames = ...
density = cs.VoxelGrid.from_stack([cs.VoxelGrid.from_file(filename) for filename in filenames])
ensemble = cs.Ensemble(density=density, pose=pose, conformation=0)
```

The stack of electron densities is stored in a single `ElectronDensity`, whose parameters now have a leading batch dimension. This can either be evaluated at a particular conformation (as in this example) or can be used across `vmap` boundaries.

Next, the model for the electron microscope. `Optics` and `Detector` models and their respective parameters are initialized. These are stored in the `Instrument` container.

```python
optics = cs.CTFOptics(defocus_u=10000.0, defocus_v=9800.0, defocus_angle=10.0)
detector = cs.GaussianDetector(variance=cs.Constant(1.0))
instrument = cs.Instrument(optics=optics, detector=detector)
```

Here, the `Detector` is simply modeled by gaussian white noise. The `CTFOptics` has all parameters used in CTFFIND4, which take their default values if not
explicitly configured here. Finally, we can instantiate the `ImagePipeline`.

```python
key = jax.random.PRNGKey(seed=0)
pipeline = cs.ImagePipeline(scattering=scattering, ensemble=ensemble, instrument=instrument)
image = pipeline.sample(key)
```

This computes an image using the noise model of the detector. One can also compute an image without the stochastic part of the model.

```python
image = pipeline.render()
```

Imaging models also accept a series of `Filter`s and `Mask`s. For example, one could add a `LowpassFilter`, `WhiteningFilter`, and a `CircularMask`.

```python
micrograph = ...  # A micrograph used for whitening
filter = cs.LowpassFilter(manager, cutoff=1.0)  # Cutoff modes above Nyquist frequency
          * cs.WhiteningFilter(manager, micrograph=micrograph)
mask = cs.CircularMask(manager, radius=1.0)     # Cutoff pixels above radius equal to (half) image size
pipeline = cs.ImagePipeline(
    scattering=scattering, ensemble=ensemble, instrument=instrument, filter=filter, mask=mask
    )
image = pipeline.sample(key)
```

`cryojax` also defines a library of `Distribution`s, which take an `ImagePipeline` as input. For example, instantiate an `IndependentFourierGaussian` distribution to call its log likelihood function.

```python
observed = ...
model = cs.IndependentFourierGaussian(pipeline)
log_likelihood = model.log_probability(observed)
```

Note that the user may need to do preprocessing of `observed`, such as applying the relevant `Filter`s and `Mask`s.

Additional components can be plugged into the image formation model. For example, modeling the solvent is supported through the `ImagePipeline`'s `Ice` model. Models for exposure to the electron beam are supported through the `Instrument`'s `Exposure` model.

For these more advanced examples, see the tutorials section of the repository. In general, `cryojax` is designed to be very extensible and new models can easily be implemented.

## Creating a loss function

In `jax`, we may want to build a loss function and apply functional transformations to it. Assuming we have already globally configured our model components at our desired initial state, the below creates a loss function at an updated set of parameters. First, we must update the model.

```python

@jax.jit
def update_model(model, params):
    """
    Update the model with equinox.tree_at (https://docs.kidger.site/equinox/api/manipulation/#equinox.tree_at).
    """
    where = lambda model: (model.pipeline.ensemble.pose.view_phi, model.pipeline.instrument.optics.defocus_u, model.pipeline.scattering.pixel_size)
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
params = dict(view_phi=jnp.asarray(jnp.pi), defocus_u=jnp.asarray(9000.0), pixel_size=jnp.asarray(density.voxel_size+0.02))
log_likelihood, grad = loss(params, model, observed)
```

To summarize, this example creates a loss function at an updated set of `Pose`, `Optics`, and `ScatteringModel` parameters. In general, any `cryojax` `Module` may contain model parameters. The exception to this is in the `ImageManager`, `Filter`, and `Mask`. These classes do computation upon initialization, so they should not be explicitly instantiated in the loss function evaluation. Another gotcha is that if the `model` is not passed as an argument to the loss, there may be long compilation times because the electron density will be treated as static. However, this may result in slight speedups.

In general, there are many ways to write loss functions. See the [equinox](https://github.com/patrick-kidger/equinox/) documentation for more use cases.

## Features

- Imaging models in `cryojax` support `jax` functional transformations, such as automatic differentiation with `grad`, paralellization with `vmap` and `pmap`, and just-in-time compilation with `jit`. Models also support GPU/TPU acceleration.
- `cryojax.Module`s, including `ImagePipeline` models, are JSON serializable thanks to the package `dataclasses-json`. The method `Module.dumps` serializes the object as a JSON string, and `Module.loads` instantiates it from the string. For example, write a model to disk with `model.dump("model.json")` and instantiate it with `cs.ImagePipeline.load("model.json")`.
- A `cryojax.Module` is just an `equinox.Module` with added serialization functionality. Therefore, the `equinox` ecosystem is available for usage!

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

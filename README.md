# Cryo-EM image simulation and analysis powered by JAX.
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

Next, [install jax-finufft](https://github.com/dfm/jax-finufft). Non-uniform FFTs are provided as an option for computing image projections. Note that this package does not yet provide GPU support, but there are plans to do so.

Finally, install `cryojax`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/cryojax
cd cryojax
python -m pip install .
```

This will install the remaining dependencies, such as [equinox](https://github.com/patrick-kidger/equinox/) for jax-friendly dataclasses, [jaxlie](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, [mrcfile](https://github.com/ccpem/mrcfile) for I/O, and [dataclasses-json](https://github.com/lidatong/dataclasses-json) for serialization.

## Building a model

Please note that this library is currently experimental and the API is subject to change! The following is a basic workflow to generate an image with a gaussian white noise model.

First, instantiate the image formation method ("scattering") and its respective representation
of an electron density ("specimen").

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs

template = "example.mrc"
scattering = cs.FourierSliceScattering(shape=(320, 320))
density = cs.ElectronGrid.from_file(template)
```

Here, `template` is a 3D electron density map in MRC format. This could be taken from the [EMDB](https://www.ebi.ac.uk/emdb/), or rasterized from a [PDB](https://www.rcsb.org/). [cisTEM](https://github.com/timothygrant80/cisTEM) provides an excellent rasterization tool in its image simulation program. In the above example, a voxel electron density in fourier space is loaded and the fourier-slice projection theorem is initialized. We can now intstantiate the biological `Specimen`.

```python
specimen = cs.Specimen(density, resolution=1.1)
```

This is a container for the parameters and metadata stored in the electron density, along with additional parameters such as the rasterization `resolution`.

Next, the model is configured for a given realization of the specimen. Here, `Pose`, `Optics`, and `Detector` models and their respective parameters are initialized. These are stored in the `PipelineState` container.

```python
key = jax.random.PRNGKey(seed=0)
pose = cs.EulerPose(view_phi=0.0, view_theta=0.0, view_psi=0.0)
optics = cs.CTFOptics(defocus_u=10000.0, defocus_v=9800.0, defocus_angle=10.0)
detector = cs.GaussianDetector(key=key, pixel_size=1.1, variance=cs.Constant(1.0))
state = cs.PipelineState(pose=pose, optics=optics, detector=detector)
```

Then, an `ImagePipeline` model is chosen. Here, we choose `GaussianImage`.

```python
model = cs.GaussianImage(scattering=scattering, specimen=specimen, state=state)
image = model()
```

Imaging models also accept a series of `Filter`s and `Mask`s. For example, one could add a `LowpassFilter`, `WhiteningFilter`, and a `CircularMask`.

```python
filters = [cs.LowpassFilter(scattering.padded_shape, cutoff=1.0),  # Cutoff modes above Nyquist frequency
           cs.WhiteningFilter(scattering.padded_shape, micrograph=micrograph)]
masks = [cs.CircularMask(scattering.shape, radius=1.0)]           # Cutoff pixels above radius equal to (half) image size
model = cs.GaussianImage(scattering=scattering, specimen=specimen, state=state, filters=filters, masks=masks)
image = model()
```

If a `GaussianImage` is initialized with the field `observed`, the model will instead compute a Gaussian log-likelihood in Fourier space with a diagonal covariance tensor (or power spectrum).

```python
model = cs.GaussianImage(scattering=scattering, specimen=specimen, state=state, observed=observed)
log_likelihood = model()
```

Note that the user may need to do preprocessing of `observed`, such as applying the relevant `Filter`s and `Mask`s.

Additional components can be plugged into the `ImagePipeline` model's `PipelineState`. For example, `Ice` and electron beam `Exposure` models are supported. For example, `GaussianIce` models the ice as gaussian noise, and `UniformExposure` multiplies the image by a scale factor. Imaging models from different stages of the pipeline are also implemented. `ScatteringImage` computes images solely with the scattering model, while `OpticsImage` uses a scattering and optics model. `DetectorImage` turns this into a detector readout, while `GaussianImage` adds the ability to evaluate a gaussian likelihood.

For these more advanced examples, see the tutorials section of the repository. In general, `cryojax` is designed to be very extensible and new models can easily be implemented.

## Creating a loss function

In `jax`, we ultimately want to build a loss function and apply functional transformations to it. Assuming we have already globally configured our model components at our desired initial state, the below creates a loss function at an updated set of parameters. First, we must build the model.

```python
import equinox as eqx

def build_model(params: dict[str, jax.Array]) -> cs.GaussianImage:
    # Perform "model surgery" with equinox.tree_at
    p = eqx.tree_at(lambda p: p.view_phi, pose, params["view_phi"])
    o = eqx.tree_at(lambda o: o.defocus_u, optics, params["defocus_u"])
    d = eqx.tree_at(lambda d: d.pixel_size, detector, params["pixel_size"])
    # Build the PipelineState
    state = cs.PipelineState(pose=p, optics=o, detector=d)
    # Build the model
    model = cs.GaussianImage(
        scattering=scattering, specimen=specimen, state=state, observed=observed
    )
    return model
```

Note that the `PipelineState` contains all of the model parameters in this example. `Specimen` can also contain model parameters. The `ElectronDensity`, `ScatteringConfig`, `Filter`s, and `Mask`s all do computation upon initialization that should not be included in the loss function evaluation. We can now create the loss!

```python
@jax.jit
@jax.value_and_grad
def loss(params: dict[str, jax.Array]) -> jax.Array:
    model = build_model(params)
    return model()
```

Finally, we can evaluate the log_likelihood at an updated set of parameters.

```python
params = dict(view_phi=jnp.asarray(jnp.pi), defocus_u=jnp.asarray(9000.0), pixel_size=jnp.asarray(1.30))
log_likelihood = loss(params)
```

To summarize, this example creates a loss function at an updated set of `Pose`, `Optics`, and `Detector` parameters. In general, there are many ways to write loss functions. See the [equinox](https://github.com/patrick-kidger/equinox/) documentation for more complex use cases.

Note that one could also write a custom log likelihood function simply by instantiating a model without the observed data.

## Features

- Imaging models in `cryojax` support `jax` functional transformations, such as automatic differentiation with `grad`, paralellization with `vmap` and `pmap`, and just-in-time compilation with `jit`. Models also support GPU/TPU acceleration. However, until GPU support for `jax-finufft` is added, models using the `NufftScattering` method will not support the GPU.
- `cryojax.Module`s, including `ImagePipeline` models, are JSON serializable thanks to the package `dataclasses-json`. The method `Module.dumps` serializes the object as a JSON string, and `Module.loads` instantiates it from the string. For example, write a model to disk with `model.dump("model.json")` and instantiate it with `cs.GaussianImage.load("model.json")`.
- A `cryojax.Module` is just an `equinox.Module` with added serialization functionality. Therefore, the entire `equinox` ecosystem is available for usage!

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

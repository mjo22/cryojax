<h1 align='center'>cryojax</h1>

![Tests](https://github.com/mjo22/cryojax/actions/workflows/testing.yml/badge.svg)
![Lint](https://github.com/mjo22/cryojax/actions/workflows/black.yml/badge.svg)

`cryojax` is a library for cryo-EM image simulation and analysis. It is built on [`jax`](https://github.com/google/jax).

## Summary

The core of this package is its ability to model image formation in cryo-EM. These models can be fed into standard sampling, optimization, and model building libraries in `jax`, such as [`blackjax`](https://github.com/blackjax-devs/blackjax), [`optimistix`](https://github.com/patrick-kidger/optimistix), or [`numpyro`](https://github.com/pyro-ppl/numpyro).

Dig a little deeper and you'll find that `cryojax` aims to be a fully extensible modeling language for cryo-EM image formation. It implements a collection of abstract interfaces, which aim to be general enough to support any level of modeling complexity—from simple linear image formation to the most realistic physical models in the field. Best of all, these interfaces are all part of the public API. Users can create their own extensions to `cryojax`, tailored to their specific use-case!

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

This will install the remaining dependencies, such as [`equinox`](https://github.com/patrick-kidger/equinox/) for jax-friendly dataclasses, [`jaxlie`](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, and [`mrcfile`](https://github.com/ccpem/mrcfile) for I/O.

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. These are included as an option for computing image projections. In this case, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Simulating an image

The following is a basic workflow to generate an image with a gaussian white noise model.

First, instantiate the electron density representation and its respective method for computing image projections.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs
from cryojax.io import read_array_with_spacing_from_mrc

# Instantiate the scattering potential.
filename = "example_scattering_potential.mrc"
real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(filename)
potential = cs.FourierVoxelGrid.from_real_voxel_grid(real_voxel_grid, voxel_size)
# ... now instantiate fourier slice extraction
shape, pixel_size = (320, 320), voxel_size
config = cs.ImageConfig(shape, pixel_size)
scattering = cs.FourierSliceExtract(config, interpolation_order=1)
```

Here, the 3D scattering potential array is read from `filename`. Then, the abstraction of the scattering potential is then loaded in fourier-space into a `FourierVoxelGrid`, and the fourier-slice projection theorem is initialized with `FourierSliceExtract`. The scattering potential can be generated with an external program, such as [cisTEM](https://github.com/timothygrant80/cisTEM).

We can now instantiate the representation of a biological specimen, which also includes a pose.

```python
# First instantiate the pose. Translations are in Angstroms, angles are in degrees
pose = cs.EulerPose(offset_x=5.0, offset_y=-3.0, view_phi=20.0, view_theta=80.0, view_psi=-10.0)
# ... now, build the biological specimen
specimen = cs.Specimen(potential, pose)
```

Next, the model for the electron microscope.

```python
from cryojax.image import operators as op

# First, initialize the CTF and its optics model
ctf = cs.CTF(defocus_u=10000.0, defocus_v=9800.0, astigmatism_angle=10.0, voltage_in_kilovolts=300.0)
optics = cs.WeakPhaseOptics(ctf, envelope=op.FourierGaussian(b_factor=5.0))  # defocus and b_factor in Angstroms and Angstroms^2, respectively
# ... now, the model for the exposure to electrons
dose = cs.ElectronDose(electrons_per_angstrom_squared=op.Constant(100.0))  # Integrated dose rate in electrons / Angstrom^2
# ... and finally, the detector
detector = cs.PoissonDetector(dqe=cs.IdealDQE())
# ... these are stored in the Instrument
instrument = cs.Instrument(optics, dose, detector)
```

Here, the `GaussianDetector` is simply modeled by gaussian white noise. The `CTF` has all parameters used in CTFFIND4, which take their default values if not
explicitly configured here. Finally, we can instantiate the `ImagePipeline` and simulate an image.

```python
# Build the image formation model
pipeline = cs.ImagePipeline(specimen, scattering, instrument)
# ... generate an RNG key and simulate
key = jax.random.PRNGKey(seed=0)
image = pipeline.sample(key)
```

This computes an image using the noise model of the detector. One can also compute an image without the stochastic part of the model.

```python
# Compute an image without stochasticity
image = pipeline.render()
```

Instead of simulating noise from the model of the detector, `cryojax` also defines a library of distributions. These distributions define the stochastic model from which images are drawn. For example, instantiate an `IndependentFourierGaussian` distribution and either sample from it or compute its log-likelihood

```python
from cryojax.image import rfftn
from cryojax.inference import distributions as dist
from cryojax.image import operators as op

# Passing the ImagePipeline and a variance function, instantiate the distribution
model = dist.IndependentFourierGaussian(pipeline, variance=op.Constant(1.0))
# ... then, either simulate an image from this distribution
key = jax.random.PRNGKey(seed=0)
image = model.sample(key)
# ... or compute the likelihood
observed = rfftn(...)  # for this example, read in observed data and take FFT
log_likelihood = model.log_probability(observed)
```

For more advanced image simulation examples and to understand the many features in this library, see the documentation (coming soon!).

## Creating a loss function

In `jax`, we may want to build a loss function and apply functional transformations to it. Assuming we have already globally configured our model components at our desired initial state, the below creates a loss function at an updated set of parameters. First, we must update the model.

```python

@jax.jit
def update_distribution(distribution, params):
    """
    Update the model with equinox.tree_at (https://docs.kidger.site/equinox/api/manipulation/#equinox.tree_at).
    """
    where = lambda model: (
        distribution.pipeline.specimen.pose.view_phi,
        distribution.pipeline.instrument.optics.ctf.defocus_u,
        distribution.pipeline.scattering.config.pixel_size
    )
    updated_distribution = eqx.tree_at(
        where, distribution, (params["view_phi"], params["defocus_u"], params["pixel_size"])
    )
    return updated_distribution
```

We can now create the loss and differentiate it with respect to the parameters.

```python
@jax.jit
@jax.value_and_grad
def loss(params, distribution, observed):
    distribution = update_distribution(distribution, params)
    return distribution.log_probability(observed)
```

Finally, we can evaluate an updated set of parameters.

```python
params = dict(
    view_phi=jnp.asarray(jnp.pi),
    defocus_u=jnp.asarray(9000.0),
    pixel_size=jnp.asarray(density.voxel_size+0.02),
)
log_likelihood, grad = loss(params, model, observed)
```

To summarize, this example creates a loss function at an updated set of parameters. In general, any `cryojax` object may contain model parameters and there are many ways to write loss functions. See the [equinox](https://github.com/patrick-kidger/equinox/) documentation for more use cases.

## Features

- Imaging models in `cryojax` support `jax` functional transformations, such as automatic differentiation with `grad`, paralellization with `vmap` and `pmap`, and just-in-time compilation with `jit`. Models also support GPU/TPU acceleration.
- `cryojax` is built on `equinox`. Therefore, the `equinox` ecosystem is available for usage! Learning `equinox` is strongly recommended.

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

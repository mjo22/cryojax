# Cryo-EM image simulation and analysis powered by JAX.
This library is a modular framework for simulating forward models of cryo electron microscopy images. It is built on [JAX](https://github.com/google/jax).

## Summary

The core of this package is its ability to simulate cryo-EM images. Starting with a 3D electron density map, one can simulate a scattering process onto the imaging plane with modulation by the instrument optics. Images are then sampled from models of the noise or the corresponding log-likelihood is computed.

These models can be fed into standard statistical inference or optimization libraries in `jax`, such as [numpyro](https://github.com/pyro-ppl/numpyro) or [jaxopt](https://github.com/google/jaxopt).

## Installation

Installing `cryojax` is currently a bit more complicated than it should be because it relies on the library [tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft) through the `jax` experimental feature [jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md). Getting [tensorflow](https://github.com/tensorflow/tensorflow) to install besides `jax` is a bit of a pain. Once [jax-finufft](https://github.com/dfm/jax-finufft) adds GPU support, the `tensorflow` dependency will be replaced. In the following instructions, please install `tensorflow` and `jax` with either GPU or CPU support.

To start, I recommend creating a new virtual environment. For example, you could do this with `conda`.

```bash
conda create -n cryojax -c conda-forge python=3.10
```

I recommend using `python=3.10` because of recent features and best practices in type checking, but all that is necessary is `python>=3.7` for [dataclasses](https://docs.python.org/3/library/dataclasses.html) support. Custom dataclasses that are safe to pass to `jax` are heavily used in this library!

Then, modify the [tensorflow installation](https://www.tensorflow.org/install/pip) instructions to install version 2.11.x. As I'm writing this, `tensorflow-nufft` supports up to version 2.11.x, which is tested on `3.7<=python<=3.10`. `tensorflow>=2.12.x` and `python>=3.11` may not be supported.

Now [install tensorflow-nufft](https://mrphys.github.io/tensorflow-nufft/guide/start/) with

```bash
pip install tensorflow-nufft==0.12.0
```

Next, [install JAX](https://github.com/google/jax#installation). I have found it easier to first install `tensorflow` and then `jax` because each try to install their own CUDA dependencies. `tensorflow` seems to be more strict about this.

Finally, install `cryojax`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/cryojax
cd cryojax
python -m pip install .
```

This will install the remaining dependencies, such as [jaxlie](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, [mrcfile](https://github.com/ccpem/mrcfile) for I/O, and [dataclasses-json](https://github.com/lidatong/dataclasses-json) for serialization.

## Usage

Please note that this library is currently experimental and the API is subject to change!

The following is a basic workflow to generate an image with a gaussian white noise model.

```python
import jax.numpy as jnp
from cryojax.utils import irfft
from cryojax.io import load_grid_as_cloud
import cryojax.simulator as cs

template = "example.mrc"
key = jax.random.PRNGKey(seed=0)
scattering = cs.NufftScattering(shape=(320, 320), pixel_size=1.32)
cloud = load_grid_as_cloud(template, config)
pose, optics, detector = cs.EulerPose(), cs.CTFOptics(), cs.WhiteNoiseDetector(key=key)
state = cs.ParameterState(pose=pose, optics=optics, detector=detector)
model = cs.GaussianImage(scattering=scattering, specimen=cloud, state=state)
params = dict(view_phi=np.pi, defocus_u=8000., alpha=1.4)
image = irfft(model(params))  # The image is returned in Fourier space.
```

Here, `template` is a 3D electron density map in MRC format. This could be taken from the [EMDB](https://www.ebi.ac.uk/emdb/), or rasterized from a [PDB](https://www.rcsb.org/). [cisTEM](https://github.com/timothygrant80/cisTEM) provides an excellent rasterization tool in its image simulation program. In the above example, a rasterzied grid is converted to a density point cloud and read into a `Cloud`. Alternatively, a user could instantiate a custom `Cloud`.

This workflow configures an initial model state at the library's default parameters, then evaulates it at a state with an updated viewing angle `view_phi`, major axis defocus `defocus_u`, and detector noise variance `alpha`. For a more advanced example, see the tutorials section of the repository (stay tuned!).

If a `GaussianImage` is initialized with the field `observed`, the model will instead compute a Gaussian log-likelihood in Fourier space with a diagonal covariance tensor (or power spectrum).

```python
from cryojax.utils import fft

model = cs.GaussianImage(scattering=scattering, specimen=cloud, state=state, observed=fft(observed))
log_likelihood = model(params)
```

Imaging models also accept a series of `Filter`s and `Mask`s. By default, this is an `AntiAliasingFilter` that cuts off modes above the Nyquist freqeuency. Alternatively, one could add a `WhiteningFilter` and a `CircularMask`.

```python
from cryojax.utils import fftfreqs

filters = [cs.AntiAliasingFilter(scattering.pixel_size * scattering.freqs, cutoff=0.667),  # Cutoff modes above 2/3 Nyquist frequency
           cs.WhiteningFilter(scattering.pixel_size * scattering.freqs, fftfreqs(micrograph.shape), micrograph)]
masks = [cs.CircularMask(scattering.coords / scattering.pixel_size, radius=1.0)]           # Cutoff pixels above radius equal to (half) image size
model = cs.GaussianImage(scattering=scattering, specimen=cloud, state=state, filters=filters, masks=masks)
image = irfft(model(params))
```

Ice models are also supported. For example, `EmpiricalIce` stores an empirical measure of the ice power spectrum. `ExponentialNoiseIce` generates ice as noise whose correlations decay exponentially. Imaging models from different stages of the pipeline are also implemented. `ScatteringImage` computes images solely with the scattering model, while `OpticsImage` uses a scattering and optics model. `DetectorImage` turns this into a detector readout, while `GaussianImage` adds the ability to evaluate a gaussian likelihood. In general, `cryojax` is designed to be very extensible and new models can easily be implemented.

## Features

Imaging models in `cryojax` support `jax` functional transformations, such as automatic differentiation with `grad` and paralellization with `vmap` and `pmap`. Models also support GPU/TPU acceleration. Until GPU support for `jax-finufft` is added, `jit` compilation will not be supported because `tensorflow-nufft` does not compile.

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

## Acknowledgments

The tooling, packaging structure, and API in `cryojax` are influenced by the library [tinygp](https://github.com/dfm/tinygp), which is also written in `jax` and makes use of custom, jax-friendly python `dataclasses`. Thank you for the developers for providing an excellent model for this package!

# 2D Template Matching of Cryo-EM images powered by JAX.
This library is an implementation of 2D template matching (2DTM) in cryo electron microscopy micrographs built on [JAX](https://github.com/google/jax). Starting with a 3D electron density map, one can simulate images with models of the scattering onto the imaging plane and the modulation by the instrument optics. Then, one can generate images sampled from models of the noise or compute a corresponding log-likelihood.

## Installation

Installing `jax-2dtm` is currently a bit more complicated than it should be because it relies on the library [tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft) through the `jax` experimental feature [jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md). Getting [tensorflow](https://github.com/tensorflow/tensorflow) to install besides `jax` is a bit of a pain. Once [jax-finufft](https://github.com/dfm/jax-finufft) adds GPU support, the `tensorflow` dependency will be replaced. In the following instructions, please install `tensorflow` or `jax` with the appropriate configuration with your workflow. Namely, install both with GPU support if you would like.

To start, I recommend creating a new virtual environment. For example, you could do this with `conda`:

```bash
conda create -n jax-2dtm -c conda-forge python=3.10
```

I recommend using `python=3.10` because of recent features and changing best practices in type checking, but really all that is necessary is support for python [dataclasses](https://docs.python.org/3/library/dataclasses.html) (`python>=3.7`). Custom dataclasses that are safe to pass to `jax` are heavily used in this library!

Then, modifiy the [tensorflow installation](https://www.tensorflow.org/install/pip) instructions to install version 2.11.x. Be careful to not install the latest version! As I'm writing this, `tensorflow-nufft` supports up to version 2.11.x, which supports `python` versions between 3.7 and 3.10. Make sure you're not using `python>=3.11`!

Finally, [install tensorflow-nufft](https://mrphys.github.io/tensorflow-nufft/guide/start/) with

```bash
pip install tensorflow-nufft==0.12.0
```

Next, [install JAX](https://github.com/google/jax#installation). I have found it easier to first install `tensorflow` and then `jax` because each try to install their own CUDA dependencies, and `tensorflow` seems to be more strict about this. It may be possible to reverse these instructions!

Finally, install `jax-2dtm`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/jax-2dtm
cd jax-2dtm
python -m pip install .
```

This will install the remaining dependencies, such as [jaxlie](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, [mrcfile](https://github.com/ccpem/mrcfile) for I/O, and [dataclasses-json](https://github.com/lidatong/dataclasses-json) for serialization.

## Usage

Please note that this library is currently experimental and the API is subject to change!

The following is a basis workflow to generate an image with a gaussian white noise model:

```python
import jax.numpy as jnp
from jax_2dtm.utils import ifft
from jax_2dtm.io import load_grid_as_cloud
from jax_2dtm.simulator import ScatteringConfig
from jax_2dtm.simulator import EulerPose, CTFOptics, WhiteNoise, Intensity, ParameterState
from jax_2dtm.simulator import GaussianImage

template = "example.mrc"
config = ScatteringConfig(shape=(320, 320), pixel_size=1.32)
cloud = load_grid_as_cloud(template, config)
pose, optics, intensity, noise = EulerPose(), CTFOptics(), Intensity(), WhiteNoise()
state = ParameterState(pose=pose, optics=optics, intensity=intensity, noise=noise)
model = GaussianImage(config=config, cloud=cloud, state=state)
params = dict(view_phi=np.pi, defocus_u=8000., sigma=1.4)
image = ifft(model(params))  # The image is returned in Fourier space.
```

Here, `template` is a 3D electron density map in MRC format. This could be taken from the [EMDB](https://www.ebi.ac.uk/emdb/), or rasterized from a [PDB](https://www.rcsb.org/). [cisTEM](https://github.com/timothygrant80/cisTEM) provides an excellent rasterization tool in its image simulation program.

This workflow configures an initial model state at the library's default parameters, then evaulates it at a state with updated `view_phi`, `defocus_angle`, and `sigma` (the white noise variance). To see a more advanced example, see the tutorials section of the repository (stay tuned!).

If a `GaussianImage` is initialized with the field `observed`, the model will instead compute a Gaussian log-likelihood with a diagonal covariance tensor in Fourier space (i.e. the power spectrum).

```python
model = GaussianImage(config=config, cloud=cloud, state=state, observed=observed)
log_likelihood = model(params)
```

Imaging models also accept a series of `Filter`s. By default, this is an `AntiAliasingFilter` that cuts off modes above the Nyquist freqeuency. Alternatively, this could be a different `AntiAliasingFilter` and a `WhiteningFilter`.

```python
from jax_2dtm.simulator import AntiAliasingFilter, WhiteningFilter

filters = [AntiAliasingFilter(config.pixel_size * config.freqs, cutoff=0.667),  # Cutoff modes above 2/3 Nyquist frequency
           WhiteningFilter(config.pixel_size * config.freqs, micrograph)]
model = GaussianImage(config=config, cloud=cloud, state=state, observed=observed, filters=filters)
image = ifft(model(params))
```

Alternative noise models are supported. For example, `EmpiricalNoise` stores an empirical covariance matrix. `LorenzianNoise` is a toy model for generating correlated noise. Imaging models from different stages of the pipeline are also implemented. `ScatteringImage` computes images solely with the scattering model, while `OpticsImage` uses the scattering and optics model.

## Features

Imaging models in `jax-2dtm` support `jax` functional transformations, such as automatic differentiation with `grad` and paralellization with `vmap` and `pmap`. Models also support GPU/TPU acceleration. Until GPU support for `jax-finufft` is added, `jit` compilation will not be supported because `tensorflow-nufft` does not compile.

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a highly successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.

__all__ = ["typing", "io", "simulator", "core", "image"]

from cryojax.cryojax_version import __version__
import importlib as _importlib


def __getattr__(name):
    return _importlib.import_module("." + name, __name__)


__author__ = "Michael O'Brien"
__email__ = "michaelobrien@g.harvard.edu"
__uri__ = "https://github.com/mjo22/cryojax"
__description__ = "Cryo-EM image simulation and analysis powered by JAX"

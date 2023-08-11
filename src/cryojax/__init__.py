__all__ = ["core", "io", "simulator", "utils"]

import importlib
from cryojax.cryojax_version import __version__


def __getattr__(name):
    return importlib.import_module("." + name, __name__)


del importlib

__author__ = "Michael O'Brien"
__email__ = "michaelobrien@g.harvard.edu"
__uri__ = "https://github.com/mjo22/cryojax"
__description__ = "Cryo-EM image simulation and analysis powered by JAX"

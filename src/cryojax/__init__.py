__all__ = ["Module", "field", "types", "io", "simulator", "utils"]

import importlib
from cryojax.cryojax_version import __version__

from .core import Module, field


def __getattr__(name):
    return importlib.import_module("." + name, __name__)


__author__ = "Michael O'Brien"
__email__ = "michaelobrien@g.harvard.edu"
__uri__ = "https://github.com/mjo22/cryojax"
__description__ = "Cryo-EM image simulation and analysis powered by JAX"

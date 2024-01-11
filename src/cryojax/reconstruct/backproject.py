from cryojax.simulator import Filter
from cryojax.typing import RealImage

__all__ = ["WeinerFilter"]


class WeinerFilter(Filter):
    def __init__(self, ctf: RealImage, noise_level: float = 0.0):
        self.filter = ctf / (ctf * ctf + noise_level)

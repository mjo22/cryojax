"""
Routines for starfile serialization and deserialization.
"""

import starfile
from pathlib import Path

from ..simulator import ImageConfig, CTF, EulerAnglePose


def read_detection_from_starfile(filename: str | Path):
    """Read a particle detection from a starfile."""
    filename = Path(filename)
    # Make sure filename is valid starfile
    _validate_filename(filename)
    # Read starfile
    star = starfile.read(filename)


def _validate_filename(filename: Path):
    suffixes = filename.suffixes
    if not (len(suffixes) == 1 and suffixes[0] == ".star"):
        raise IOError(f"Filename should include .star suffix. Got filename {filename}.")

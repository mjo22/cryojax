"""
Routines for starfile serialization and deserialization.
"""

import starfile
import pathlib
from typing import Any
from os import PathLike


def read_and_validate_starfile(filename: PathLike, **kwargs: Any):
    """Read a particle detection from a starfile."""
    # Make sure filename is valid starfile
    _validate_filename(filename)
    # Read starfile
    return starfile.read(filename, **kwargs)


def _validate_filename(filename: PathLike):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == ".star"):
        raise IOError(
            "Tried to read STAR file, but the filename does not include a .star suffix. "
            f"Got filename '{filename}'."
        )

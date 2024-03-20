"""
Routines for starfile serialization and deserialization.
"""

import starfile
import pathlib
import pandas as pd
from typing import Any, cast


def read_and_validate_starfile(filename: str | pathlib.Path) -> dict[str, pd.DataFrame]:
    """Read a particle detection from a starfile."""
    # Make sure filename is valid starfile
    _validate_filename(filename)
    # Read starfile
    return cast(
        dict[str, pd.DataFrame], starfile.read(pathlib.Path(filename), always_dict=True)
    )


def _validate_filename(filename: str | pathlib.Path):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == ".star"):
        raise IOError(
            "Tried to read STAR file, but the filename does not include a .star suffix. "
            f"Got filename '{filename}'."
        )

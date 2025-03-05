"""
Objects for reading PDB files based on MDTraj's PDBTrajectoryFile class.

This module is based on the MDTraj library, which is distributed under the
LPGL-2.1 license. The original source code can be found at
https://github.com/mdtraj/mdtraj

"""

import gzip
import importlib.resources
import pathlib
import xml.etree.ElementTree as etree
from copy import copy
from io import StringIO
from typing import Optional
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative
from urllib.request import urlopen

import mdtraj
import numpy as np
from mdtraj.core import element as elem
from mdtraj.core.topology import Topology
from mdtraj.formats.pdb.pdbfile import PDBTrajectoryFile
from mdtraj.formats.pdb.pdbstructure import PdbStructure
from mdtraj.utils import open_maybe_zipped


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


def _is_url(url):
    """Check to see if a URL has a valid protocol.
    from pandas/io.common.py Copyright 2014 Pandas Developers
    Used under the BSD licence
    """
    try:
        return urlparse(url).scheme in _VALID_URLS
    except (AttributeError, TypeError):
        return False


class PDBReader:
    """Read a PDB file and return the positions, bfactors, atomic numbers
    and topology. This class is based on MDtraj's PDBTrajectoryFile class.

    **Attributes:**

    - `positions`: The cartesian coordinates of all of the atoms in each frame.
    - `topology`: mdtraj.core.Topology, default=None
        if you give a topology as input the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files
    - `bfactors`: The B-factors of all of the atoms in each frame.
    - `atomic_numbers`: The atomic numbers of all of the atoms in each frame.
    - `unitcell_lengths`: The unitcell lengths (3-tuple) in this PDB file. May be None
    - `unitcell_angles`: The unitcell angles (3-tuple) in this PDB file. May be None
    """

    distance_unit = "angstroms"
    _residueNameReplacements = {}
    _atomNameReplacements = {}
    _chain_names = [chr(ord("A") + i) for i in range(26)]

    def __init__(
        self,
        filename: str | pathlib.Path,
        is_assembly: bool = False,
        i_model: Optional[int] = None,
        standard_names: bool = True,
        top: Optional[Topology] = None,
    ):
        """Open a PDB file for reading.

        **Arguments:**

        - `filename`: The name of the PDB file to open. Can be a URL.
        - `is_assemble`: If the pdb file contains multiple models, set this to `True`.
                Warning: if your pdb is a trajectory, all frames will be loaded.
        - `i_model`: Index of the returned mode.
            Should only be used if `is_assemble` is `True`.
        - `standard_names` : bool, default=True
            If True, non-standard atomnames and residuenames are standardized to conform
            with the current PDB format version. If set to false, this step is skipped.
        - `top` : mdtraj.core.Topology, default=None
            if you give a topology as input the topology won't be parsed from the pdb file
            it saves time if you have to parse a big number of files
        """

        if i_model is not None and not is_assembly:
            raise ValueError(
                "i_model should only be used when assembling a biological assembly"
            )

        self._open = False
        self._file = None
        self._topology = top
        self._positions = None
        self._bfactors = None
        self._atomic_numbers = None
        self._last_topology = None
        self._standard_names = standard_names
        self._is_assemble = is_assembly
        self._i_model = i_model

        PDBReader._loadNameReplacementTables()

        if _is_url(filename):
            self._file = urlopen(filename)
            if filename.lower().endswith(".gz"):
                self._file = gzip.GzipFile(fileobj=self._file)
            self._file = StringIO(self._file.read().decode("utf-8"))

        else:
            self._file = open_maybe_zipped(filename, "r")

        self._read_models()
        self._open = True

    @classmethod
    def set_chain_names(cls, values):
        """Set the cycle of chain names used when writing PDB files

        When writing PDB files, PDBTrajectoryFile translates each chain's
        index into a name -- the name is what's written in the file. By
        default, chains are named with the letters A-Z.

        Parameters
        ----------
        values : list
            A list of chacters (strings of length 1) that the PDB writer will
            cycle through to choose chain names.
        """
        for item in values:
            if not isinstance(item, str) and len(item) == 1:
                raise TypeError("Names must be a single character string")
        cls._chain_names = values

    @property
    def positions(self):
        """The cartesian coordinates of all of the atoms in each frame.
        Available when a file is opened in mode='r'
        """
        return self._positions

    @property
    def bfactors(self):
        """The B-factors of all of the atoms in each frame.
        Available when a file is opened in mode='r'
        """
        return self._bfactors

    @property
    def atomic_numbers(self):
        """The atomic numbers of all of the atoms in each frame.
        Available when a file is opened in mode='r'
        """
        return self._atomic_numbers

    @property
    def topology(self):
        """The topology from this PDB file.
        Available when a file is opened in mode='r'
        """
        return self._topology

    @property
    def unitcell_lengths(self):
        "The unitcell lengths (3-tuple) in this PDB file. May be None"
        return self._unitcell_lengths

    @property
    def unitcell_angles(self):
        "The unitcell angles (3-tuple) in this PDB file. May be None"
        return self._unitcell_angles

    @property
    def closed(self):
        "Whether the file is closed"
        return not self._open

    def close(self):
        "Close the PDB file"
        if self._open:
            self._file.close()
        self._open = False

    def _read_models(self):
        pdb = PdbStructure(self._file, load_all_models=True)

        if len(pdb) > 1 and not self._is_assemble:
            raise ValueError(
                "PDB Error: The PDB file contains multiple models. "
                "Use 'assemble=True' to build a biological assembly."
                "Loading trajectories is currently not supported."
            )

        _positions = []
        _bfactors = []
        _atomic_numbers = []

        # load all of the positions (from every model)
        for i, model in enumerate(pdb.iter_models(use_all_models=True)):
            if self._i_model is not None and i != self._i_model:
                continue
            for chain in model.iter_chains():
                for residue in chain.iter_residues():
                    for atom in residue.atoms:
                        _positions.append(atom.get_position())
                        _bfactors.append(atom.get_temperature_factor())
                        _atomic_numbers.append(atom.element.atomic_number)

        self._positions = np.array(_positions)
        self._bfactors = np.array(_bfactors)
        self._atomic_numbers = np.array(_atomic_numbers)

        ## The atom positions read from the PDB file
        self._unitcell_lengths = pdb.get_unit_cell_lengths()
        self._unitcell_angles = pdb.get_unit_cell_angles()

        # Load the topology if None is given
        if self._topology is None:
            self._topology = Topology()

            atomByNumber = {}
            for i, model in enumerate(pdb.iter_models(use_all_models=True)):
                if self._i_model is not None and i != self._i_model:
                    continue
                for chain in model.iter_chains():
                    c = self._topology.add_chain(chain.chain_id)
                    for residue in chain.iter_residues():
                        resName = residue.get_name()
                        if (
                            resName in PDBTrajectoryFile._residueNameReplacements
                            and self._standard_names
                        ):
                            resName = PDBTrajectoryFile._residueNameReplacements[resName]
                        r = self._topology.add_residue(
                            resName,
                            c,
                            residue.number,
                            residue.segment_id,
                        )
                        if (
                            resName in PDBTrajectoryFile._atomNameReplacements
                            and self._standard_names
                        ):
                            atomReplacements = PDBTrajectoryFile._atomNameReplacements[
                                resName
                            ]
                        else:
                            atomReplacements = {}
                        for atom in residue.atoms:
                            atomName = atom.get_name()
                            if atomName in atomReplacements:
                                atomName = atomReplacements[atomName]
                            atomName = atomName.strip()
                            element = atom.element
                            if element is None:
                                element = PDBTrajectoryFile._guess_element(
                                    atomName,
                                    residue.name,
                                    len(residue),
                                )

                            newAtom = self._topology.add_atom(
                                atomName,
                                element,
                                r,
                                serial=atom.serial_number,
                                formal_charge=atom.formal_charge,
                            )
                            atomByNumber[atom.serial_number] = newAtom

            self._topology.create_standard_bonds()
            self._topology.create_disulfide_bonds(self.positions)

            # Add bonds based on CONECT records.
            connectBonds = []
            for model in pdb.models:
                for connect in model.connects:
                    i = connect[0]
                    for j in connect[1:]:
                        if i in atomByNumber and j in atomByNumber:
                            connectBonds.append((atomByNumber[i], atomByNumber[j]))
            if len(connectBonds) > 0:
                # Only add bonds that don't already exist.
                existingBonds = {
                    (bond.atom1, bond.atom2) for bond in self._topology.bonds
                }
                for bond in connectBonds:
                    if (
                        bond not in existingBonds
                        and (bond[1], bond[0]) not in existingBonds
                    ):
                        self._topology.add_bond(bond[0], bond[1])
                        existingBonds.add(bond)

    @staticmethod
    def _loadNameReplacementTables():
        """Load the list of atom and residue name replacements."""
        if len(PDBTrajectoryFile._residueNameReplacements) == 0:
            tree = etree.parse(
                importlib.resources.files(mdtraj) / "formats/pdb/data/pdbNames.xml",
            )
            allResidues = {}
            proteinResidues = {}
            nucleicAcidResidues = {}
            for residue in tree.getroot().findall("Residue"):
                name = residue.attrib["name"]
                if name == "All":
                    PDBTrajectoryFile._parseResidueAtoms(residue, allResidues)
                elif name == "Protein":
                    PDBTrajectoryFile._parseResidueAtoms(residue, proteinResidues)
                elif name == "Nucleic":
                    PDBTrajectoryFile._parseResidueAtoms(residue, nucleicAcidResidues)
            for atom in allResidues:
                proteinResidues[atom] = allResidues[atom]
                nucleicAcidResidues[atom] = allResidues[atom]
            for residue in tree.getroot().findall("Residue"):
                name = residue.attrib["name"]
                for id in residue.attrib:
                    if id == "name" or id.startswith("alt"):
                        PDBTrajectoryFile._residueNameReplacements[residue.attrib[id]] = (
                            name
                        )
                if "type" not in residue.attrib:
                    atoms = copy(allResidues)
                elif residue.attrib["type"] == "Protein":
                    atoms = copy(proteinResidues)
                elif residue.attrib["type"] == "Nucleic":
                    atoms = copy(nucleicAcidResidues)
                else:
                    atoms = copy(allResidues)
                PDBTrajectoryFile._parseResidueAtoms(residue, atoms)
                PDBTrajectoryFile._atomNameReplacements[name] = atoms

    @staticmethod
    def _guess_element(atom_name, residue_name, residue_length):
        "Try to guess the element name"

        upper = atom_name.upper()
        if upper.startswith("CL"):
            element = elem.chlorine
        elif upper.startswith("NA"):
            element = elem.sodium
        elif upper.startswith("MG"):
            element = elem.magnesium
        elif upper.startswith("BE"):
            element = elem.beryllium
        elif upper.startswith("LI"):
            element = elem.lithium
        elif upper.startswith("K"):
            element = elem.potassium
        elif upper.startswith("ZN"):
            element = elem.zinc
        elif residue_length == 1 and upper.startswith("CA"):
            element = elem.calcium

        # TJL has edited this. There are a few issues here. First,
        # parsing for the element is non-trivial, so I do my best
        # below. Second, there is additional parsing code in
        # pdbstructure.py, and I am unsure why it doesn't get used
        # here...
        elif residue_length > 1 and upper.startswith("CE"):
            element = elem.carbon  # (probably) not Celenium...
        elif residue_length > 1 and upper.startswith("CD"):
            element = elem.carbon  # (probably) not Cadmium...
        elif residue_name in ["TRP", "ARG", "GLN", "HIS"] and upper.startswith("NE"):
            element = elem.nitrogen  # (probably) not Neon...
        elif residue_name in ["ASN"] and upper.startswith("ND"):
            element = elem.nitrogen  # (probably) not ND...
        elif residue_name == "CYS" and upper.startswith("SG"):
            element = elem.sulfur  # (probably) not SG...
        else:
            try:
                element = elem.get_by_symbol(atom_name[0])
            except KeyError:
                try:
                    symbol = (
                        atom_name[0:2].strip().rstrip("AB0123456789").lstrip("0123456789")
                    )
                    element = elem.get_by_symbol(symbol)
                except KeyError:
                    element = None

        return element

    @staticmethod
    def _parseResidueAtoms(residue, map):
        for atom in residue.findall("Atom"):
            name = atom.attrib["name"]
            for id in atom.attrib:
                map[atom.attrib[id]] = name

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __len__(self):
        "Number of frames in the file"
        if not self._open:
            raise ValueError("I/O operation on closed file")
        return len(self._positions)

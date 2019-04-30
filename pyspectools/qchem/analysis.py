
from dataclasses import dataclass, field
from typing import List, Dict
import os
from glob import glob
from copy import deepcopy
import shutil
import tempfile

import numpy as np
from matplotlib import pyplot as plt

from pyspectools.qchem import utils, parsers
from pyspectools import units, routines
from pyspectools import figurefactory as ff


@dataclass
class CalculationSet:
    """
    A master handler of a set of calculations. This class works in two ways: to manage a bunch of calculations
    that together yield a closed set of thermochemistry, or a set of calculations that sum up to a single
    composite value.
    """
    paths: List = field(default_factory=list)
    species: List = field(default_factory=list)
    total_energy: float = 0.
    energies: np.array = np.array([])
    converted_energies: np.array = np.array([])
    relative_energies: np.array = np.array([])
    calculations: Dict = field(default_factory=dict)

    @classmethod
    def from_folder(cls, path, extension="log", program="g16"):
        if program == "g16":
            parser = CalculationResult.from_g16
        calculations = dict()
        species = list()
        # Sorting the list means it should work platform independent, since glob returns a list
        # of files in a different order depending on the OS...
        paths = sorted(
            list(
                glob(os.path.join(path, "*{}".format(extension)))
            )
        )
        for index, filepath in enumerate(paths):
            filename = os.path.splitext(filepath)[0].split("/")[-1]
            calculations[index] = parser(filepath)
            species.append(calculations[index].smi)
        calc_set = cls(paths=paths, species=species, calculations=calculations)
        return calc_set

    @classmethod
    def from_pkl(cls, filepath):
        calc_set = routines.read_obj(filepath)
        return calc_set

    def __post_init__(self):
        for folder in ["png"]:
            try:
                os.mkdir(folder)
            except FileExistsError:
                pass

    def __add__(self, other):
        return self.total_energy - other.total_energy

    def __sub__(self, other):
        return self.total_energy - other.total_energy

    def compare_energies(self, level="composite", conversion=None):
        # If there are no energies set up yet, try to get them from the individual calculations
        if len(self.energies) == 0:
            self.energies = np.array([calc.__getattribute__(level) for index, calc in self.calculations.items()])
        self.relative_energies = self.energies - self.energies.min()
        if conversion:
            if conversion not in ["wavenumber", "kJ/mol", "eV", "K"]:
                raise Exception("{} unit not implemented.".format(conversion))
            else:
                if conversion == "wavenumber":
                    conv_func = units.hartree2wavenumber
                elif conversion == "kJ/mol":
                    conv_func = units.hartree2kjmol
                elif conversion == "eV":
                    conv_func = units.haev
                elif conversion == "K":
                    conv_func = units.hak
            # Convert the relative energies into the specified units
            self.converted_energies = conv_func(self.relative_energies)

    def sum_energies(self, level="composite"):
        """
        Calculates the total energy of all the species in this CalculationSet.

        Parameters
        ----------
        level: str
            Specifies the level of theory to make the comparison.

        Returns
        -------
        total_energy: float
            The total energy in Hartrees
        """
        if len(self.energies) == 0:
            self.energies = np.array([calc.__getattribute__(level) for index, calc in self.calculations.items()])
        self.total_energy = np.sum(self.energies)
        return self.total_energy

    def create_portrait(self, **kwargs):
        """
        Create a collage of 2D depictions of each of the species within this set of data. Uses matplotlib imshow
        to show the images inline in a Jupyter notebook.

        Parameters
        ----------
        kwargs
            Additional keywords are passed to the subplots generation

        Returns
        -------
        fig, axarray
            Matplotlib figure and axis arrays
        """
        _ = [calc.to_png() for index, calc in self.calculations.items()]
        png_names = ["png/{}.png".format(calc.filename) for index, calc in self.calculations.items()]
        images = [plt.imread(file) for file in png_names]
        nrows = int(len(png_names) / 3)
        ncols = 3
        fig, axarray = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        for index, pkg in enumerate(zip(images, axarray.reshape(-1))):
            image, ax = pkg
            ax.imshow(image)
            ax.set_title = "Molecule index {}".format(index)
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)
        return fig, axarray

    def copy_species(self, indexes):
        """
        Spawn a copy of the CalculationSet, only with the specified species in the set.

        Parameters
        ----------
        indexes: list of int
            A list of indexes, corresponding to the species to be kept in the new copy

        Returns
        -------
        new_instance
            A deepcopy of this CalculationSet with only the specified species
        """
        new_instance = deepcopy(self)
        new_instance.calculations = {
            index: calc for index, calc in new_instance.calculations if index in indexes
        }
        return new_instance

    def create_pes(self, width=5., x=None, **kwargs):
        """
        Create a 
        Parameters
        ----------
        width
        x
        kwargs

        Returns
        -------

        """
        if x is None:
            x = np.arange(len(self.energies))
        if len(self.converted_energies) != 0:
            y = self.converted_energies
        elif len(self.relative_energies) != 0:
            y = self.relative_energies
        else:
            y = self.energies
        pes_x, pes_y = ff.make_pes(x, y, width)

        fig, ax = plt.subplots(**kwargs)
        ax.plot(pes_x, pes_y)

        return fig, ax


@dataclass
class CalculationResult:
    """
    Class for handling individual calculations. The idea behind this class is to be as general as possible,
    handling the data that get parsed out by specific program parsers defined in qchem.parsers.
    """
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    success: bool = False
    u_A: float = 0.0
    u_B: float = 0.0
    u_C: float = 0.0
    id: int = 0
    formula: str = ""
    smi: str = ""
    program: str = ""
    point_group: str = "C1"
    method: str = ""
    basis: str = ""
    charge: int = 0
    multi: int = 1
    kappa: float = 0.0
    DJ: float = 0.0
    DJK: float = 0.0
    DK: float = 0.0
    delJ: float = 0.0
    delK: float = 0.0
    Iaa: float = 0.0
    Ibb: float = 0.0
    Icc: float = 0.0
    defect: float = 0.0
    scf: float = 0.0
    cc: float = 0.0
    correlation: float = 0.0
    dboc: float = 0.0
    rel: float = 0.0
    coords: str = ""
    zpe: float = 0.0
    elec_zpe: float = 0.0
    composite: float = 0.0
    ts: bool = False
    harm_freq: List[float] = field(default_factory=list)
    harm_int: List[float] = field(default_factory=list)
    harm_sym: List[str] = field(default_factory=list)
    anharm_freq: List[float] = field(default_factory=list)
    anharm_int: List[float] = field(default_factory=list)
    anharm_dipole: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    opt_delta: float = 0.0
    type: str = "scf"
    filename: str = ""

    @classmethod
    def from_yml(cls, filepath):
        """
        Class method for reading in a YAML calculation dump created with the to_yml method.

        Parameters
        ----------
        filepath - str
            Filepath to the YAML file.

        Returns
        -------
        Calculation - object
            Instance of the Calculation object from parsing the YAML file.
        """
        data = routines.read_yaml(filepath)
        return cls(**data)

    @classmethod
    def from_g16(cls, filepath):
        """
        Class method for parsing a Gaussian logfile, and converting the dictionary into a Calculation class.

        Parameters
        ----------
        filepath - str
            Filepath to the Gaussian output file.

        Returns
        -------
        Calculation - object
            Calculation object with the parsed Gaussian output.
        """
        data = parsers.parse_g16(filepath)
        return cls(**data)

    def __add__(self, other):
        for attr in ["composite", "elec_zpe", "correlation", "scf"]:
            if hasattr(self, attr):
                key = attr
                break
        try:
            new_instance = deepcopy(self)
            new_instance.smi += " + {}".format(other.smi)
            new_instance.formula += " + {}".format(other.formula)
            new_instance.__dict__[key] += other.__dict__[key]
            return new_instance
        except AttributeError:
            raise Exception("No valid energy value found: {}".format(key))

    def __sub__(self, other):
        for attr in ["composite", "elec_zpe", "correlation", "scf"]:
            if hasattr(self, attr):
                key = attr
                break
        try:
            new_instance = deepcopy(self)
            new_instance.smi = new_instance.smi.replace(" + {}".format(other.smi), "")
            new_instance.formula = new_instance.formula.replace(" + {}".format(other.formula), "")
            new_instance.__dict__[key] -= other.__dict__[key]
            return new_instance
        except AttributeError:
            raise Exception("No valid energy value found: {}".format(key))

    def __repr__(self):
        return self.smi

    def __str__(self):
        return self.smi

    def to_yml(self, filepath):
        """
        Function to dump the Calculation to disk in YAML syntax.

        Parameters
        ----------
        filepath - str
            Filepath to the YAML file target.
        """
        routines.dump_yaml(filepath, self.__dict__)

    def to_png(self):
        """
        Generate a PNG file by dumping the SMI to a temporary folder, converting, and copying back to the current
        working directory using obabel.
        Returns
        -------

        """
        if shutil.which("obabel") is None:
            raise Exception("obabel executable not found in path.")
        curdir = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)
            with open("{}.smi".format(self.filename), "w+") as write_file:
                write_file.write(self.smi)
            utils.obabel_png("{}.smi".format(self.filename), "smi")
            shutil.copy2("{}.png".format(self.filename), os.path.join(curdir, "png/{}.png".format(self.filename)))
            os.chdir(curdir)

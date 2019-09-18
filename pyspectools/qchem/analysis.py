
from dataclasses import dataclass, field
from typing import List, Dict
import os
from glob import glob
from copy import deepcopy
import shutil
import tempfile
import warnings

import numpy as np
from matplotlib import pyplot as plt
import periodictable
from uncertainties import ufloat

from pyspectools.qchem import utils, parsers, multiwell
from pyspectools import units, routines
from pyspectools import figurefactory as ff


@dataclass
class CalculationSet:
    """
    A master handler of a set of calculations. This class works in two ways: to manage a bunch of calculations
    that together yield a closed set of thermochemistry, or a set of calculations that sum up to a single
    composite value.
    """
    paths: List[str] = field(default_factory=list)
    species: List = field(default_factory=list)
    total_energy: float = 0.
    atoms: str = ""
    energies: np.array = np.array([])
    converted_energies: np.array = np.array([])
    relative_energies: np.array = np.array([])
    calculations: Dict = field(default_factory=dict)
    products: List = field(default_factory=list)
    reactants: List = field(default_factory=list)

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
        for folder in ["png", "outputs"]:
            try:
                os.mkdir(folder)
            except FileExistsError:
                pass

    def __add__(self, other):
        if self != other:
            warnings.warn(
                "Atomic composition between the two sets are not the same! {} != {}".format(
                    self._eval_formula(),
                    other._eval_formula()
                )
            )
        return self.total_energy - other.total_energy

    def __sub__(self, other):
        if self != other:
            warnings.warn(
                "Atomic composition between the two sets are not the same! {} != {}".format(
                    self.eval_formula(),
                    other.eval_formula()
                )
            )
        return self.total_energy - other.total_energy

    def __eq__(self, other):
        current_formula = self.eval_formula()
        other_formula = other.eval_formula()
        return current_formula == other_formula

    def eval_formula(self):
        current_formula = "".join([calc.formula for index, calc in self.calculations.items()])
        current_formula = periodictable.formula(current_formula).atoms
        return current_formula

    def compare_energies(self, index=None, level="composite", conversion=None):
        # If there are no energies set up yet, try to get them from the individual calculations
        if len(self.energies) == 0:
            self.energies = np.array([calc.__getattribute__(level) for index, calc in self.calculations.items()])
        if index is None:
            comp = min(self.energies)
        else:
            comp = self.energies[index]
        self.relative_energies = self.energies - comp
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
        names = [calc.filename for index, calc in self.calculations.items()]
        images = [plt.imread(file) for file in png_names]
        nrows = int(len(png_names) / 3) + 1
        ncols = 3
        fig, axarray = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        for index, pkg in enumerate(zip(images, names, axarray.reshape(-1))):
            image, name, ax = pkg
            ax.imshow(image)
            ax.set_title("{} - {}".format(name, index))
        for ax in axarray.reshape(-1):
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
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
        new_instance = CalculationSet()
        new_instance.calculations = {
            index: calc for index, calc in self.calculations.items() if index in indexes
        }
        new_smi = [calc.smi for index, calc in self.calculations.items() if index in indexes]
        new_instance.species = [smi for smi in self.species if smi in new_smi]
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

    def save(self, filepath):
        """
        Dump the current CalculationSet to disk as a pickle file.

        Parameters
        ----------
        filepath: str
            Path to the file you wish to save to.
        """
        routines.save_obj(self, filepath)


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
    anharm_zpe: float = 0.0
    elec_zpe: float = 0.0
    composite: float = 0.0
    ts: bool = False
    harm_freq: List[float] = field(default_factory=list)
    harm_int: List[float] = field(default_factory=list)
    harm_sym: List[str] = field(default_factory=list)
    anharm_freq: List[float] = field(default_factory=list)
    anharm_int: List[float] = field(default_factory=list)
    anharm_dipole: List[float] = field(default_factory=list)
    thermo_corrections: Dict = field(default_factory=dict)
    alphas: List[float] = field(default_factory=list)
    G3: Dict = field(default_factory=dict)
    opt_delta: float = 0.0
    type: str = "scf"
    filename: str = ""
    image_path: str = ""

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
    def from_g16(cls, filepath, parser=parsers.parse_g16):
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
        data = parser(filepath)
        calc_obj = cls()
        calc_obj.__dict__.update(**data)
        return calc_obj

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
        """
        curdir = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)
            with open("{}.smi".format(self.filename), "w+") as write_file:
                write_file.write(self.smi)
            utils.obabel_png("{}.smi".format(self.filename), "smi")
            shutil.copy2("{}.png".format(self.filename), os.path.join(curdir, "png/{}.png".format(self.filename)))
            os.chdir(curdir)

    def to_xyz(self):
        if self.program == "Gaussian":
            format = "g09"
        else:
            format = "smi"
        utils.obabel_xyz(self.filename, format=format)

    def to_multiwell(self, state="reac", comment="", delh=0.):
        """
        Convert the parsed results into Multiwell format. Wraps the function
        from the `multiwell` module, `format_multiwell`.

        Parameters
        ----------
        state: str, optional
            Specify which part of the reaction this structure is.
        comment: str, optional
            Comment line for the structure
        delh: float, optional
            The 0 K heat of formation, or relative energy of this point in the units
            specified in the multiwell input file (usually kJ/mol)

        Returns
        -------
        mw_str: str
            String containing the formatted output
        """
        return multiwell.format_multiwell(self.__dict__, state, comment, delh)

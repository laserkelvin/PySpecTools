"""
Scripts to perform rate calculations using Multiwell.
The focus here is to take output from a pseudo-variational
calculation performed using CFOUR, format the results (e.g.
constants, frequencies, etc.) into ktools format
"""

import os
from subprocess import Popen, PIPE
import logging
from glob import glob

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as cf
from matplotlib import pyplot as plt

from pyspectools.qchem import parsers
from pyspectools.routines import read_yaml


class Vtst:
    """
    Class that handles a VTST calculation. This class is mainly
    designed for filepath handling, since the multiwell programs
    are extremely annoying about relative paths...
    """
    @classmethod
    def vtst_calc(cls, calc_root, reaction, ktools_yml_path):
        """ Serialization method specifically for VTST calculations
        using ktools.

        The required inputs are:
        calc_root - path to root of the calculation directory
        reaction - str specification to name the reaction
        ktools_yml_path - path to ktools YAML input
        """
        vtst_obj = cls(calc_root, reaction, ktools_yml_path)
        # Get all the filepaths to calculations
        vtst_obj.logs = glob(
            os.path.join(
                vtst_obj.paths["calc_root"],
                "*/freq*.log"
                )
            )
        # Initialize ktools settings
        vtst_obj.init_ktools()
        return vtst_obj

    @classmethod
    def tst_calc(cls, data_dict, reaction, ktools_yml_path):
        """ Serialization method for conventional TST calculations
        using ktools.

        The required inputs are:
        data_dict - dictionary holding paths to output files, with
        the keys corresponding to reactant, ts, product as well as
        reaction enthalpies
        reaction - str specification to name the reaction
        ktools_yml_path - path to ktools YAML input

        format of data_dict should be like this:
        data_dict = {"reactant": {"path": , "E"}}
        """
        tst_obj = cls(None, reaction, ktools_yml_path)
        # Convert all paths to absolute ones
        tst_obj.logs = dict()
        for key in data_dict:
            tst_obj.logs[key] = os.path.abspath(
                data_dict[key]["path"]
                )
        tst_obj.data = data_dict.copy()
        # Initialize ktools settings
        tst_obj.init_ktools()
        return tst_obj


    def __init__(self, calc_root, reaction, ktools_yml_path):
        """ Init method for Ktools calculation
        Depending on the type of calculation - either TST or VTST -
        serialize the Ktools instance with those specific methods.
        """
        # Dictionary for holding all of the paths
        self.paths = dict()
        # Dictionary for holding all of the template strings
        self.temps = dict()
        """
        Set up paths management
        """
        # Path to notebook directory
        self.paths["nb"] = os.getcwd()
        # Path to multiwell templates
        self.paths["mw_temp"] = os.path.join(
            self.paths["nb"],
            "multiwell"
            )
        # Path to the calculation root. Only used for VTST
        if calc_root is not None:
            self.paths["calc_root"] = os.path.abspath(calc_root)
        # If output folder is not present, make it
        if os.path.isdir("ktools-vtst") is False:
            os.mkdir("ktools-vtst")
        # output_path is where the ktools logs will be stored
        self.paths["output"] = os.path.join("ktools-vtst", reaction)
        if os.path.isdir(self.paths["output"]) is False:
            os.mkdir(self.paths["output"])
        # Convert all paths into absolute paths
        for key, value in self.paths.items():
            self.paths[key] = os.path.abspath(value)
            print(self.paths[key])
        """
        Read in ktools settings
        """
        self.ktools = read_yaml(ktools_yml_path)
        for key, value in self.ktools.items():
            print(key + ":\t\t" + str(value))
        """
        Reading in the string templates
        """
        print("Found the following templates:")
        for temp_path in glob(os.path.join(self.paths["mw_temp"], "*.temp")):
            with open(temp_path) as read_file:
                self.temps[os.path.basename(temp_path)] = read_file.read()
                print(temp_path)

    def init_ktools(self):
        # sort the list of calculations
        self.ktools["NT"] = len(self.ktools["Tlist"])
        self.ktools["Tlist"] = [str(T) for T in self.ktools["Tlist"]]
        self.ktools["Tlist"] = " ".join(self.ktools["Tlist"])
        self.ktools["Nreact"] = "1"
        self.ktools["Nprod"] = "1"
        self.ktools["NTS"] = len(self.logs) - 2

    def analyze_tst(self):
        """ Analysis function for transition state RRKM
        More or less the same as the VTST version, but simpler
        """
        os.chdir(self.paths["output"])
        mol_str = ""
        for species in self.data:
            sym_path = os.path.basename(self.logs[species])
            try:
                os.symlink(
                    self.logs[species],
                    sym_path
                    )
            except FileExistsError:
                pass
            # Parse log file
            parsed_data = parsers.parse_cfour(sym_path)
            # Series of if cases to determine what the formatting
            # should be
            if species == "reactant":
                species_type = "reac"
                name = "Reactant"
                dist = 0.
            elif species == "product":
                species_type = "prod"
                name = "Product"
                dist = 2.
            elif species == "ts":
                species_type = "ctst"
                name = "TS"
                dist = 1.
            # Create a species object with all its associated
            # parameters
            current_species = Species(
                self.temps["mominert.temp"],
                name,
                parsed_data,
                species_type,
                dist,
                self.data[species]["E"]
                )
            mol_str += current_species.format_species() + "\n"
        self.ktools["molecules"] = mol_str
        self.dump_ktools()
        print("Dumped ktools input file to disk")
        output, error = self.call_ktools(
                os.path.join(self.paths["output"], "ktools.inp")
                )
        os.chdir(self.paths["nb"])
        return output, error

    def analyze_vtst(self):
        """ Main analysis function
        Loops over all of the log files detected in __init__ and
        produces ktools input file format strings
        """
        os.chdir(self.paths["output"])
        mol_str = ""
        end_step = self.ktools["stepsize"] * self.ktools["steps"]
        irc_values = np.arange(0, end_step, self.ktools["stepsize"])
        steps = list()
        energies = list()
        # Loop over the log files and generate Species objects
        for index in range(len(self.logs)):
            # Create symlinks to the logfile
            log_path = os.path.join(
                self.paths["calc_root"],
                str(index) + "/freq" + str(index) + ".log"
                )
            sym_path = os.path.basename(log_path)
            try:
                os.symlink(
                    log_path,
                    sym_path
                    )
            except FileExistsError:
                pass
            # Parse the log file
            parsed_data = parsers.parse_cfour(sym_path)
            print("Step " + str(index))
            # First step is always global minimum
            if index == 0:
                species_type = "reac"
                name = "Reactant"
                E0 = parsed_data["total energy"] + parsed_data["zpe"] / 2625.50
                E = 0.
            # If it's the last species, treat as products
            elif index == len(self.logs) - 1:
                species_type = "prod"
                name = "Product"
                current_E = parsed_data["total energy"] + parsed_data["zpe"] / 2625.5
                E = (current_E - E0) * 2625.50
            else:
                # Everything else is trial TS
                species_type = "ctst"
                name = "TS-" + str(index + 1)
                current_E = parsed_data["total energy"] + parsed_data["zpe"] / 2625.5
                E = (current_E - E0) * 2625.50
            dist = np.round(irc_values[index], decimals=2)
            E = np.round(E, decimals=3)
            current_species = Species(
                self.temps["mominert.temp"],
                name,
                parsed_data,
                species_type,
                dist,
                E
                )
            mol_str += current_species.format_species() + "\n"
            steps.append(dist)
            energies.append(E)
        # Add the complete molecule specification
        self.ktools["molecules"] = mol_str
        # Write the ktools file to disk
        self.dump_ktools()
        #print("Calling ktools executable")
        #output, error =self.call_ktools(
        #    os.path.join(self.paths["output"], "ktools.inp")
        #    )
        os.chdir(self.paths["nb"])
        return steps, energies

    def dump_ktools(self):
        # Method for writing the ktools input file to disk
        # Open file and write ktools input
        with open(os.path.join(self.paths["output"], "ktools.inp"), "w+") as write_file:
            write_file.write(
                self.temps["ktools.temp"].format_map(self.ktools)
                )

    def call_ktools(self, inp_path):
        # wrapper for ktools executable
        with Popen(["ktools", inp_path], stdout=PIPE) as ktools_exe:
            output, error = ktools_exe.communicate()
        print(output)
        print(error)
        return output, error


class Species:
    # Class that handles an individual species in a ktools calculation
    # The required input are:
    # name: string representing name of molecule
    # log_path: string path to CFOUR calculation output
    # species_type: string designation to denote reactant, TS, product
    def __init__(self, mominert_str, name, parsed_data, species_type, dist, delh=0.):
        self.mom_temp_str = mominert_str
        #self.logger = init_logger(ktools_path + "/" + name + "-ktools.log")
        self.cfour_calc = parsed_data
        # Format parameters for mominert calculation
        geom_str = ""
        # atom_counter required because dummy atoms exist
        atom_counter = 0
        for atom in self.cfour_calc["coordinates"]:
            if atom[0] != "X":
                atom_counter += 1
                # Insert atom index into list
                atom.insert(1, atom_counter)
                # Convert float values for coordinates to string
                atom = [str(item) for item in atom]
                # Flatten list to string
                atom_coords = " ".join(atom)
                # Only take real atoms
                geom_str += atom_coords + "\n"
            # Convert float values for coordinates to string
        self.path = name + "-coords"
        self.mom_dict = {
            "title": name,
            "natoms": atom_counter,
            "geometry": geom_str
        }
        # Calculate moments of inertia
        self.inertia, self.rotcon = self.calc_mominert()
        # Determine number of rotational degrees of freedom
        if self.inertia[0] == 0.:
            nrot = 1
        # For all non-linear molecules, we'll have j and krotors
        else:
            nrot = 2
        # Set up dictionary for formatting ktools input
        self.species_dict = {
            "type": species_type,
            "name": name,
            "delh": str(delh),
            "dist": str(dist),
            "formula": self.get_formula(),
            "comment": "-".join([name, species_type, "r" + str(dist)]),
            "dof": str(len(self.cfour_calc["frequencies"]) + nrot),
            "Eelev": "0.0",
            "gele": str(self.cfour_calc["multiplicity"]),
            "modes": self.format_dof()
            }

    def calc_mominert(self):
        # Write the mominert input to disk and run mominert
        with open(self.path + ".dat", "w+") as write_file:
            write_file.write(
                self.mom_temp_str.format_map(self.mom_dict)
                )
        inertia, rotcon = self.call_mominert()
        return inertia, rotcon

    def call_mominert(self):
        # Function that will call mominert to calculate moments of
        # inertia and parse the output
        with Popen(["mominert", self.path + ".dat"], stdout=PIPE) as mominert:
            output, error = mominert.communicate()
        log_file = self.path + ".out"
        # Parse the output of mominert
        inertia, rotcon = parse_mominert(log_file)
        return inertia, rotcon

    def determine_rot(self, mode_num):
        # Function to determine what kind of model to treat rotation based
        # on the output of mominert
        # Only input is the mode number; make it the last mode
        # Format string for ktools
        rot_str = "{mode} {flag} {AAA} {BBB} {CCC}"
        full_rot = ""
        # Linear molecule case
        if self.inertia[0] == 0.:
            jrotor = rot_str.format_map(
                    {
                    "mode": str(mode_num),
                    "flag": "jro",
                    "AAA": str(self.inertia[-1]),
                    "BBB": "1",
                    "CCC": "2",                # 2-fold degenerate
                    }
                )
            full_rot = jrotor
        # Asymmetric rotor
        else:
            krotor = rot_str.format_map(
                {
                "mode": str(mode_num),
                "flag": "kro",
                "AAA": self.inertia[0],    # Depends on A
                "BBB": "1",
                "CCC": "1",                # 1-fold degenerate
                }
                )
            jrotor = rot_str.format_map(
                {
                "mode": str(mode_num + 1),
                "flag": "jro",
                "AAA": np.average(self.inertia[1:]),
                "BBB": "1",
                "CCC": "2",                # 2-fold degenerate
                }
                )
            full_rot = krotor + "\n" + jrotor
        return full_rot

    def get_formula(self):
        # Function for determining chemical formula very roughly
        formula_str = ""
        for line in self.cfour_calc["coordinates"]:
            if line[0] != "X":
                formula_str += line[0]
        return formula_str

    def format_dof(self):
        # Function for formatting the internal degrees of freedom for this
        # species
        template_str = '''{mode} vib {frequency} 0 1\n'''
        freq_str = ""
        counter = 0
        for index, frequency in enumerate(self.cfour_calc["frequencies"]):
            freq_str += template_str.format_map(
                {"mode": str(index + 1), "frequency": str(frequency)}
                )
            counter = index
        # Tack on the rotational term
        freq_str += self.determine_rot(counter + 2)
        return freq_str

    def format_species(self):
        # Master function for yielding a complete specification for ktools
        # input
        temp_string = """{type} {name} {delh} {dist}
{formula}
1. {comment}
2.
3.
1 1 1
{Eelev} {gele}
{dof} har amua
{modes}
        """
        return temp_string.format_map(self.species_dict)



def init_logger(log_name):
    """
    Use `logging` module to provide debug data for HEAT analysis.
    All of the extraneous data (e.g. extraplation coefficients, etc)
    will be output with the logger.

    Required input is the name of the log file including file extension.
    Returns a logger object.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/" + log_name)
    fh.setLevel(logging.DEBUG)

    # Set up the formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

""" 
    Data parsing from the Multiwell programs
"""

def parse_mominert(momout_path):
    # Function to parse the output of a mominert calculation
    # Ordering of the values are A, B, C
    with open(momout_path) as read_file:
        lines = read_file.readlines()
    for line in lines:
        # Read in the moments of inertia in units of amu ang**2
        if "amu" in line:
            split_line = line.split()
            inertia = [
                float(split_line[2]),
                float(split_line[5]),
                float(split_line[8])
                ]
        # Read in rotational constants in MHz
        if "GHz" in line:
            split_line = line.split()
            if "Infinity" in line:
                # If molecule is linear, Ia is 0 and we will not
                # parse infinity!
                rotcon = [
                    0.,
                    float(split_line[5]),
                    float(split_line[8]),
                    ]
            else:
                rotcon = [
                    float(split_line[2]),
                    float(split_line[5]),
                    float(split_line[8])
                    ]
    # Make sure ordering of values are descending
    # Rotational constants are sorted in descending order, while
    # the moments of inertia are reversed!
    inertia = sorted(inertia, reverse=False)
    rotcon = sorted(rotcon, reverse=True)
    return inertia, rotcon


def parse_ktools(ktoolslog_path):
    """ ktools_path corresponds to the logfile for the ktools
    calculation.

    The routine will parse canonical rate constants from the
    file with extension .canonical, and J-resolved microcanonical
    constants from .kej
    """
    # Get the directory where the log files exist
    basedir = os.path.dirname(
        os.path.abspath(ktoolslog_path)
        )
    filedict = dict()
    for extension in ["*.canonical", "*.kej"]:
        target = os.path.join(basedir, extension)
        filedict[extension.split(".")[-1]] = glob(target)[0]

    analysis_dict = dict()
    # loop over the list of files now
    for name, path in filedict.items():
        with open(path) as read_file:
            # Determine which parser to use
            if name == "canonical":
                func = parse_canonical
            elif name == "kej":
                func = parse_kej
            analysis_dict[name] = func(read_file)


def parse_canonical(filecontents):
    """ Function to parse data out of a .canonical file from ktools
    Returns a dataframe containing the recommended thermal rate constants
    along with the temperature and inverse temperature.
    """
    read_flag = False
    data = list()
    read = False
    for line in filecontents:
        if read_flag is True:
            if "------" in line and read is True:
                read = False
            if read is True:
                read_out = []
                for value in line.split()[1:]:
                    # Try and convert values to flots
                    try:
                        read_out.append(np.float(value))
                    # This will fail for extremely small numbers
                    # where the ktools output screws up the formatting
                    except ValueError:
                        read_out.append(0.)
                data.append(np.array(read_out))
            if "------" in line and read is False:
                read = True
        if "FINAL RECOMMENDED REACTION RATE" in line:
            read_flag = True
    df = pd.DataFrame(data, columns=["T", "k(forward)", "k(reverse)", "Keq"])
    df["1000 / T"] = 1000. / df["T"]
    return df


"""
    Rate analysis routines
"""

def mod_arrhenius(T, a, b, g):
    # Modified Arrhenius equation from Wakelam 2010
    return a * (T / 300.)**b * np.exp(-g / T)


def fit_arrhenius(canonical_df, guess=None):
    """ Function to fit a modified Arrhenius rate model to
    a dataframe containing canonical rates extracted from
    the parse_canonical routine
    """
    # Some random initial parameters
    canonical_df = canonical_df.dropna()
    if guess is None:
        p0 = [1e-5, 0., 1e4]
    else:
        p0 = guess
    popt, pcov = cf(
            mod_arrhenius,
            canonical_df["T"],
            canonical_df["k(forward)"],
            p0=p0
            )
    results = {
            "a": popt[0],
            "b": popt[1],
            "g": popt[2]
            }

    fig, ax = plt.subplots()

    ax.plot(
            canonical_df["T"],
            canonical_df["k(forward)"],
            lw=2,
            label="Data"
            )

    ax.plot(
            canonical_df["T"],
            mod_arrhenius(canonical_df["T"], *p0),
            linestyle="--",
            alpha=0.7,
            label="Initial"
            )
    ax.plot(
            canonical_df["T"],
            mod_arrhenius(canonical_df["T"], **results),
            linestyle="--",
            label="Fit"
            )

    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")

    return results


def format_multiwell(molecule, state="reac", comment="", delh=0.):
    """
    Convert the output of an electronic structure program into the form
    used for a lot of the Multiwell programs; specifically for `thermo`.

    Takes the parsed output as a dictionary, and the user can specify
    what state the structure is (e.g. reac, ctst, or prod).

    Parameters
    ----------
    molecule: dict
        Dictionary containing parsed output from electronic structure programs.
        The format is that generated by the CalculationResult class.
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
    mw_str = """{state} {name} {delh}
{formula}
! {comment}
!
!
1 1 1
0. {multi}
{ndof} 'HAR' 'MHZ'
{vibrot}
    """
    vibrot = ""
    # Round the frequencies to the nearest wavenumber
    molecule["harm_freq"] = np.unique(np.round(molecule["harm_freq"], 0))[::-1]
    ndof = len(molecule["harm_freq"]) + 2
    for index, freq in enumerate(molecule["harm_freq"]):
        vibrot += "{:d} vib {:.0f} 0. 1\n".format(index + 1, freq)
    vibrot += "{:d} qro {:.3f} 1\n".format(index + 2, molecule["A"])
    vibrot += "{:d} qro {:.3f} 2\n".format(index + 3, molecule["B"])
    form_dict = {
        "vibrot": vibrot,
        "ndof": ndof,
        "multi": molecule["multi"],
        "comment": comment,
        "formula": molecule["formula"],
        "state": state,
        "name": molecule["filename"],
        "delh": delh
    }
    return mw_str.format(**form_dict)

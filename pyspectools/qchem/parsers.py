
"""

parsers.py

This module contains routines for parsing data out of electronic structure outputs. Everything will be centered around
a Calculation dataclass, with specific parsing methods written for each program. The result should be a homogenized
Pythonization of calculations, regardless of program used.

"""

from typing import List, Dict
from dataclasses import dataclass, field

import periodictable
import numpy as np

from pyspectools.routines import dump_yaml, read_yaml


@dataclass
class CalculationResult:
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
    correlation: float = 0.0
    dboc: float = 0.0
    rel: float = 0.0
    coords: str = ""
    zpe: float = 0.0
    Etot: float = 0.0
    harm_freq: List[float] = field(default_factory=list)
    harm_int: List[float] = field(default_factory=list)
    harm_sym: List[str] = field(default_factory=list)
    anharm_freq: List[float] = field(default_factory=list)
    anharm_int: List[float] = field(default_factory=list)
    anharm_dipole: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    opt_delta: float = 0.0
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
        data = read_yaml(filepath)
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
        data = parse_g16(filepath)
        return cls(**data)

    def to_yml(self, filepath):
        """
        Function to dump the Calculation to disk in YAML syntax.

        Parameters
        ----------
        filepath - str
            Filepath to the YAML file target.
        """
        dump_yaml(filepath, self.__dict__)


def parse_g16(filepath):
    """
    Parse in the output of a Gaussian 16 calculation. To optimize the output format, make sure the calculation
    route includes the Output=Pickett keyword, which ensures the coordinates are actually in the principal axis
    orientation.

    Parameters
    ----------
    filepath - str
        Filepath to Gaussian logfile

    Returns
    -------

    """
    data = dict()
    harm_freq = list()
    harm_int = list()
    data["program"] = "Gaussian"
    filename = filepath.split("/")[-1].split(".")[0]
    with open(filepath) as read_file:
        lines = read_file.readlines()
        for index, line in enumerate(lines):
            if "Rotational constants (MHZ)" in line:
                rot_con = lines[index + 1].split()
                rot_con = [float(value) for value in rot_con]
                A, B, C = rot_con
                data["A"] = A
                data["B"] = B
                data["C"] = C
            if "Dipole moment (Debye)" in line:
                dipoles = lines[index + 1].split()[:3]
                dipoles = [float(value) for value in dipoles]
                u_A, u_B, u_C = dipoles
                data["u_A"] = u_A
                data["u_B"] = u_B
                data["u_C"] = u_C
            if "Full point group" in line:
                data["point_group"] = line.split()[3]
            if "Stationary point found" in line:
                data["success"] = True
            if line.startswith(" # "):
                calc = line.split()[1].split("/")
                method, basis = calc
                data["method"] = method
                data["basis"] = basis
            if "Multiplicity" in line:
                split_line = line.split()
                data["charge"] = int(split_line[2])
                data["multi"] = int(split_line[-1])
            if "Vibro-Rot alpha Matrix" in line:
                alpha_flag = True
                alpha_lines = lines[index + 3:]
                alpha_mat = list()
                alpha_index = 0
                while alpha_flag is True:
                    current_line = alpha_lines[alpha_index]
                    if current_line.startswith("Q("):
                        alpha = alpha_lines[alpha_index].split()[2:]
                        alpha = [float(value) for value in alpha]
                        alpha_mat.append(alpha)
                        alpha_index += 1
                    else:
                        alpha_flag = False
                        data["alphas"] = alpha_mat
            if "Anharmonic Infrared Spectroscopy" in line:
                anharm_flag = True
                anharm_index = 0
                anharm_lines = lines[index + 9:]
                anharm_freq = list()
                anharm_int = list()
                while anharm_flag is True:
                    current_line = anharm_lines[anharm_index].split()
                    anharm_index += 1
                    if len(current_line) > 0:
                        anharm_freq.append(float(current_line[2]))
                        anharm_int.append(float(current_line[-1]))
                    else:
                        anharm_flag = False
                        data["anharm_freq"] = anharm_freq
                        data["anharm_int"] = anharm_int
            if "Electric dipole : Fundamental Bands" in line:
                anharm_dipole_flag = True
                anharm_dipole_index = 0
                anharm_lines = lines[index + 3:]
                anharm_dipoles = list()
                while anharm_dipole_flag is True:
                    current_line = anharm_lines[anharm_dipole_index].replace("D", "E").split()
                    anharm_dipole_index += 1
                    if len(current_line) > 0:
                        # Has the annoying D formatting from Fortran
                        anharm_dipoles.append([float(value) for value in current_line[1:]])
                    else:
                        data["anharm_dipole"] = anharm_dipoles
                        anharm_dipole_flag = False
            if "Asymm. param." in line:
                data["kappa"] = float(line.split()[-1])
            if "DELTA J  :" in line:
                data["DJ"] = float(line.replace("D", "E").split()[-1])
            if "DELTA JK :" in line:
                data["DJK"] = float(line.replace("D", "E").split()[-1])
            if "DELTA K  :" in line:
                data["DK"] = float(line.replace("D", "E").split()[-1])
            if "delta J  :" in line:
                data["delJ"] = float(line.replace("D", "E").split()[-1])
            if "delta K  :" in line:
                data["delK"] = float(line.replace("D", "E").split()[-1])
            if "Iaa" in line:
                split_line = line.replace("D", "E").split()
                data["Iaa"] = float(split_line[2])
                data["Ibb"] = float(split_line[4])
                data["Icc"] = float(split_line[-1])
                data["defect"] = data["Icc"] - data["Iaa"] - data["Ibb"]
            if "Principal axis orientation" in line:
                coord_lines = lines[index + 5:]
                coord_flag = True
                coord_mat = list()
                coord_index = 0
                while coord_flag is True:
                    current_line = coord_lines[coord_index]
                    if "------" in current_line:
                        coord_flag = False
                    else:
                        coords = current_line.split()[1:]
                        coords = [float(value) for value in coords]
                        coord_mat.append(coords)
                        coord_index += 1
                data["coords"] = np.array(coord_mat)
            if "Zero-point correction" in line:
                data["zpe"] = float(line.split()[2])
            if "Sum of electronic and zero-point" in line:
                data["Etot"] = float(line.split()[-1])
            if "Frequencies --" in line:
                freq = line.split()[2:]
                freq = [float(value) for value in freq]
                harm_freq.extend(freq)
            if "IR Inten" in line:
                inten = line.split()[3:]
                inten = [float(value) for value in inten]
                harm_int.extend(inten)
            if "Predicted change in Energy=" in line:
                data["opt_delta"] = float(line.replace("D", "E").split("=")[-1])
    if "coords" in data:
        atom_dict = dict()
        for coord in data["coords"]:
            element = periodictable.elements[coord[0]]
            if element not in atom_dict:
                atom_dict[element] = 1
            else:
                atom_dict[element] += 1
        molecule_string = "".join(["{}{}".format(key, value) for key, value in atom_dict.items()])
        data["formula"] = molecule_string
    data["filename"] = filename
    data["harm_freq"] = harm_freq
    data["harm_int"] = harm_int
    return data


def parseG3(filepath):
    """
    Function for parsing a G3 calculation from a Gaussian output file. This function does not generate an object,
    and instead returns all of the data as a dictionary.

    Parameters
    ----------
    filepath - str
        Filepath to the G3 Gaussian output file.

    Returns
    -------
    results - dict
        Dictionary containing the G3 output

    """
    results = dict()
    frequencies = list()
    rotational_constants = None
    with open(filepath) as read_file:
        for line in read_file:
            if "Rotational constants" in line:
                split_line = line.split()
                rotational_constants = [float(value) for value in split_line[3:]]
            if "Frequencies --" in line:
                split_line = line.split()
                frequencies.extend([float(value) for value in split_line[2:]])
            if "G3(0 K)" in line:
                split_line = line.split()
                results["G3-H-0 K"] = float(split_line[2])
            if "G3 Enthalpy" in line:
                split_line = line.split()
                results["G3-H-298 K"] = float(split_line[2])
                results["G3-S-298 K"] = float(split_line[-1])
    if len(frequencies) != 0:
        # Round the vibrational frequencies to integers
        frequencies = np.array(frequencies).astype(int)
        frequencies = frequencies[frequencies >= 0.]
        results["Frequencies"] = frequencies
    if rotational_constants is not None:
        results["Rotational constants"] = np.array(rotational_constants)
    return results

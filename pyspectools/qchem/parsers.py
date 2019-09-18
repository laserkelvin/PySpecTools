
"""

parsers.py

This module contains routines for parsing data out of electronic structure outputs. Everything will be centered around
a Calculation dataclass, with specific parsing methods written for each program. The result should be a homogenized
Pythonization of calculations, regardless of program used.

"""

import os
import shutil
import datetime

import periodictable
import numpy as np

from pyspectools.qchem import utils


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
    g3 = dict()
    w1 = dict()
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
                # If there's only one word, then it's a composite scheme without a specific basis
                if len(calc) == 1:
                    method = calc
                    basis = None
                else:
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
                data["elec_zpe"] = float(line.split()[-1])
            if "Frequencies --" in line:
                freq = line.split()[2:]
                freq = [float(value) for value in freq]
                harm_freq.extend(freq)
            if "IR Inten" in line:
                inten = line.split()[3:]
                inten = [float(value) for value in inten]
                harm_int.extend(inten)
            if "Total Anharm  " in line:
                data["anharm_zpe"] = float(line.split()[5])
            if "imaginary frequencies ignored" in line:
                data["ts"] = True
            if "Predicted change in Energy=" in line:
                data["opt_delta"] = float(line.replace("D", "E").split("=")[-1])
            # This section parses out the G3 contributions
            if "E(QCISD(T" in line:
                split_line = line.split()
                g3["QCISD(T)"] = float(split_line[1])
                g3["Empirical"] = float(split_line[-1])
            if "DE(Plus)" in line:
                split_line = line.split()
                g3["DE(Plus)"] = float(split_line[1])
                g3["DE(2df)"] = float(split_line[-1])
            if "Delta-G3" in line:
                split_line = line.split()
                g3["G3-contribution"] = float(split_line[1])
            if "G3(0 K)" in line:
                data["composite"] = float(line.split()[2])
                g3["G3-energy"] = float(line.split()[-1])
            if "G3 Enthalpy" in line:
                g3["G3-enthalpy"] = float(line.split()[2])
                g3["G3-entropy"] = float(line.split()[-1])
            if "W1BD (0 K)=" in line:
                split_line = line.split()
                w1["W1BD-H-0 K"] = float(split_line[3])
                data["composite"] = float(split_line[3])
            if "W1BD  Enthalpy" in line:
                split_line = line.split()
                w1["W1BD-H-298 K"] = float(split_line[2])
                w1["W1BD-S-298 K"] = float(split_line[-1])
            if "W1BD  Electronic Energy" in line:
                split_line = line.split()
                w1["W1BD-Electronic"] = float(split_line[-1])
    if "coords" in data:
        atom_dict = dict()
        for coord in data["coords"]:
            element = periodictable.elements[coord[0]]
            if element not in atom_dict:
                atom_dict[element] = 1
            else:
                atom_dict[element] += 1
        molecule_string = "".join(["{}{}".format(key, value) for key, value in atom_dict.items()])
        molecule_string = molecule_string.replace("1", "")
        # Delete the 1s, because they're not normal
        data["formula"] = molecule_string
    if shutil.which("obabel"):
        data["smi"] = utils.obabel_smi(filepath, format="g09")
    data["filename"] = filename
    data["harm_freq"] = harm_freq
    data["harm_int"] = harm_int
    data["G3"] = g3
    data["W1"] = w1
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

def parseW1(filepath):
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
            if "W1BD (0 K)=" in line:
                split_line = line.split()
                results["W1BD-H-0 K"] = float(split_line[3])
                results["composite"] = float(split_line[3])
            if "W1BD  Enthalpy" in line:
                split_line = line.split()
                results["W1BD-H-298 K"] = float(split_line[2])
                results["W1BD-S-298 K"] = float(split_line[-1])
            if "W1BD  Electronic Energy" in line:
                split_line = line.split()
                results["W1BD-Electronic"] = float(split_line[-1])
            if "E(ZPE)=" in line:
                split_line = line.split()
                results["ZPE"] = float(split_line[1])
    if len(frequencies) != 0:
        # Round the vibrational frequencies to integers
        frequencies = np.array(frequencies).astype(int)
        frequencies = frequencies[frequencies >= 0.]
        results["Frequencies"] = frequencies
    if rotational_constants is not None:
        results["Rotational constants"] = np.array(rotational_constants)
    return results


def parse_cfour(filepath):
    # Function that will parse the output file of a CFOUR calculation.
    # Everything is stored into dictionary items, and the entire output
    # of the calculation is stored as a single string.
    timestamp = os.path.getmtime(filepath)
    InfoDict = {
        "filename": " ",
        "basis": " ",
        "success": False,
        "method": " ",
        "dipole": [0., 0., 0.],
        "quadrupole": dict(),
        "rotational constants": [0., 0., 0.],
        "point group": " ",
        "orbitals": {
            "alpha": list(),
            "beta": list()
        },
        "scf energy": 0.,
        "ccsd energy": 0.,
        "ccsd(t) energy": 0.,
        "ccsdt energy": 0.,
        "ccsdt(q) energy": 0.,
        "total energy": 0.,
        "ene_iterations": {
            "scf_cycles": list(),
            "cc_cycles": list(),
        },
        "dboc": 0.,
        "relativistic": 0.,
        "coordinates": [],
        "input zmat": [],
        "final zmat": [],
        "frequencies": [],
        "centrifugal distortion": dict(),
        "zpe": 0.,
        "natoms": 0,
        "nscf": 0.,
        "ncc": 0.,
        "avg_scf": 0.,
        "avg_cc": 0.,
        "gradient norm": [],
        "timestamp": datetime.datetime.fromtimestamp(timestamp).strftime(
        '%Y-%m-%d %H:%M:%S.%f'
        )
    }
    scf_iter = 0
    scf_cycle = 0
    cc_cycle = 0
    geo_counter = 0
    skip_counter = 0
    CurrentCoords = []
    DipoleFlag = False
    AlphaFlag = True
    RotFlag = False
    SCFFlag = False
    CCFlag = False
    OrbFlag = False
    FreqFlag = False
    IZMATFlag = False        # Initial ZMAT file
    FZMATFlag = False        # Final ZMAT file
    ReadCoords = False
    CDFlag = False           # Centrifugal distortion terms
    read_props = False
    read_charge = False
    read_dboc = False
    read_mvd2 = False
    with open(filepath, "r") as read_file:
        for index, line in enumerate(read_file):
            if ("The final electronic energy is") in line:
                ReadLine = line.split()
                InfoDict["total energy"] = float(ReadLine[5])
                InfoDict["success"] = True
            if ("The full molecular point group ") in line:
                ReadLine = line.split()
                InfoDict["point group"] = ReadLine[6]
            if ("BASIS=") in line:
                ReadLine = line.split("=")
                InfoDict["basis"] = utils.clean_string(ReadLine[1].split()[0])
            if ("CALC_LEVEL") in line:
                ReadLine = line.split("=")
                InfoDict["method"] = ReadLine[1].split()[0]
            if ("EXCITE=") in line:
                ReadLine = line.split("=")
                InfoDict["method"] += "-" + ReadLine[1].split()[0]
            if RotFlag is True:            # if flagged to read the rotational constants
                ReadLine = line.split()
                for index, value in enumerate(ReadLine):
                    InfoDict["rotational constants"][index] = value
                RotFlag = False
            if ("Rotational constants (in MHz)") in line:
                RotFlag = True
            if FZMATFlag is True or IZMATFlag is True:
                if ("********") in line:
                    skip_counter += 1
                elif skip_counter == 1:
                    temp_zmat.append(line)
                elif skip_counter == 2:
                    skip_counter = 0
                    if FZMATFlag is True:
                        InfoDict["final zmat"] = temp_zmat
                    if IZMATFlag is True:
                        InfoDict["input zmat"] = temp_zmat
                    FZMATFlag = False
                    IZMATFlag = False
            if ("Final ZMATnew file") in line:
                temp_zmat = list()
                skip_counter = 0
                FZMATFlag = True
            if ("Input from ZMAT") in line:
                temp_zmat = list()
                skip_counter = 0
                IZMATFlag = True
            if ReadCoords is True:
                if ("----------") in line:
                    skip_counter += 1
                elif skip_counter == 1:
                    ReadLine = line.split()
                    CurrentCoords.append([ReadLine[0],
                                          float(ReadLine[2]) * 0.5291,
                                          float(ReadLine[3]) * 0.5291,
                                          float(ReadLine[4]) * 0.5291]
                                         )
                elif skip_counter == 2:
                    InfoDict["coordinates"] = CurrentCoords
                    ReadCoords = False
                    CurrentCoords = []
                    skip_counter = 0
            if ("Coordinates (in bohr)") in line:
                skip_counter = 0
                ReadCoords = True
#                if ("Conversion factor used") in line:
#                    self.InfoDict["dipole moment"] = Dipole
#                    DipoleFlag = False
            if DipoleFlag is True:
                ReadLine = line.split()
                Dipole = [float(ReadLine[2]),
                          float(ReadLine[5]),
                          float(ReadLine[8])
                          ]
                Dipole = [value * 2.54174691 for value in Dipole]
                InfoDict["dipole moment"] = Dipole
                DipoleFlag = False
#                if ("au             Debye") in line:
            if ("Components of electric dipole moment") in line:
                Dipole = [0., 0., 0.]
                DipoleFlag = True
            if ("Molecular gradient norm") in line:
                ReadLine = line.split()
                InfoDict["gradient norm"].append(float(ReadLine[3]))
                geo_counter += 1
            if ("IMULTP") in line:
                ReadLine = line.split()
                InfoDict["multiplicity"] = int(ReadLine[2])
            if OrbFlag is True:
                """ Read in orbital information """
                if ("++++++") in line:
                    OrbFlag = False
                    AlphaFlag = not AlphaFlag
                    skip_counter = 0
                if skip_counter == 1:
                    try:
                        ReadLine = line.split()
                        OrbitalNo = int(ReadLine[0])
                        Orbital = []
                        Orbital.append(float(ReadLine[2]))
                        Orbital.append(ReadLine[5])
                        Orbital.append(ReadLine[6])
                        if AlphaFlag is True:
                            InfoDict["orbitals"]["alpha"].append(Orbital)
                        else:
                            InfoDict["orbitals"]["beta"].append(Orbital)
                    except:
                        pass
                if ("----") in line:
                    skip_counter += 1
            if ("MO #        E(hartree)") in line:
                OrbFlag = True
                skip_counter = 0
            if ("Zero-point energy") in line:   # All in single line
                FreqFlag = False
                ZPE = float(line.split()[5])
            # Harmonic frequency parsing
            if FreqFlag is True:
                # Exception for imaginary frequencies which
                # will not convert to a float
                try:
                    # Get the frequency value
                    freq = line.split()[1]
                    # Ignore zero frequencies
                    if "0.0000" not in freq:
                        InfoDict["frequencies"].append(float(freq))
                except ValueError:
                    pass
            if ("Rotationally projected") in line:
                FreqFlag = True
            # ZPE parsing
            if ("Zero-point energy") in line:   # All in single line
                FreqFlag = False
                InfoDict["zpe"] = float(line.split()[5])
            """
            The following energy parsing is when the CC program used
            is the ECC routines.
            """
            if "CCSD correlation energy" in line:
                InfoDict["ccsd energy"] = float(line.split()[3])
            if "CCSD(T) correlation energy" in line:
                InfoDict["ccsd(t) energy"] = float(line.split()[3])
            if "HF-SCF" in line:
                InfoDict["scf energy"] = float(line.split()[2])
            if "E(SCF)" in line:
                try:
                    InfoDict["scf energy"] = float(line.split()[2])
                except ValueError:
                    InfoDict["scf energy"] = float(line.split()[1])
            """
            The following parsers will work for the VCC routines, which
            have slightly different formatting.
            """
            if "The reference energy" in line:
                InfoDict["scf energy"] = float(line.split()[4])
            if "E(SCF)" in line:
                line = line.replace("D", "E")
                InfoDict["scf energy"] = float(line.split()[2])
            if "The correlation energy is" in line:
                InfoDict["ccsd(t) energy"] = float(line.split()[4])
            if "E(CCSD(T))" in line:
                InfoDict["ccsd(t) energy"] = float(line.split()[2])
            if "E(CCSD)" in line:
                InfoDict["ccsd energy"] = float(line.split()[2])
            if CDFlag is True:
                if skip_counter == 2:
                    CDFlag = False
                else:
                    split_line = line.split()
                    if len(split_line) <= 2:
                        skip_counter += 1
                    else:
                        InfoDict["centrifugal distortion"][reduction][split_line[0]] = float(split_line[1])
            if "A-reduced centrifugal" in line or "S-reduced centrifugal" in line:
                if "A-reduced centrifugal" in line:
                    reduction = "A"
                elif "S-reduced centrifugal" in line:
                    reduction = "S"
                skip_counter = 0
                InfoDict["centrifugal distortion"][reduction] = dict()
                CDFlag = True
            if "Electrostatic potential at atomic centers" in line:
                read_props = False
                read_charge = False
            if read_charge is True:
                if "In kHz, Mass number" in line:
                    atom_mass = line.split()[4]
                    quadrupole_mat = np.zeros((3,3))
                if "CHIxx" in line:
                    quadrupole_mat[0,0] = float(line.split()[2])
                if "CHIyy" in line:
                    quadrupole_mat[1,1] = float(line.split()[2])
                if "CHIzz" in line:
                    quadrupole_mat[2,2] = float(line.split()[2])
                if "CHIxy" in line:
                    quadrupole_mat[0,1] = float(line.split()[2])
                if "CHIxz" in line:
                    quadrupole_mat[0,2] = float(line.split()[2])
                if "CHIyz" in line:
                    quadrupole_mat[1,2] = float(line.split()[2])
                    read_charge = False
                    InfoDict["quadrupole"][atom_number][atom_mass] = quadrupole_mat
            if read_props is True:
                if "Z-matrix center" in line:
                    read_charge = True
                    # Get the Z-matrix atom number
                    atom_number = line.split()[-1].split(":")[0]
                    InfoDict["quadrupole"][atom_number] = dict()
            if "the correlated density matrix" in line:
                read_props = True
            if read_dboc is True:
                if "The total diagonal Born-Oppenheimer correction (DBOC)" in line:
                    if "a.u." in line:
                        InfoDict["dboc"] = float(line.split()[7])
            if "Summary of diagonal Born-Oppenheimer correction at Hartree-Fock level" in line:
                read_dboc = True
            """
            Scalar relativistic effects are read in separately, and if up to
            second order is calculated, the reported value is MVD1 + MVD2.
            """
            if "Total MVD1 correction to energy:" in line:
                InfoDict["relativistic"] = float(line.split()[5])
            if read_mvd2 is True:
                if "Hartree" in line:
                    read_mvd2 = False
                    InfoDict["relativistic"] += float(line.split()[0])
            if "Two-electron darwin energy" in line:
                read_mvd2 = True
            if "Total CCSDT[Q] energy" in line:
                InfoDict["ccsdt(q) energy"] = float(line.split()[4])
            if "Total CCSDT energy [au]" in line:
                InfoDict["ccsdt energy"] = float(line.split()[4])

    # Convert the XYZ coordinates into SMILES using OBabel
    with open("coords.xyz", "w+") as write_file:
        write_file.write(str(len(InfoDict["coordinates"])) + "\n")
        write_file.write(" \n")
        for line in InfoDict["coordinates"]:
            write_file.write(" ".join(map(str, line)) + "\n")
    if shutil.which("obabel"):
        try:
            InfoDict["smiles"] = utils.xyz2smi("coords.xyz")
        except FileNotFoundError:
            print("OBabel executable not found in PATH.")
    return InfoDict


"""
This is a crudely written class for parsing
output from the electronic structure package "CFOUR".

The idea behind this set of routines is to be able to
quickly extract all of the useful information out of
a CFOUR calculation. The information is stored as a Python
dictionary, which has the syntax of JSON i.e. portable.

The idea is to also be able to generate quick HTML
reports so that I can file them into Jupyter notebooks.

Some of the code is written pythonically, while
others are not so much...
"""

import numpy as np
import os
import datetime
from .utils import xyz2smi

def clean_string(string):
    symbols = ['(', ')', '"', '\n']
    for symbol in symbols:
        string = string.replace(symbol, '')
    return string


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
                InfoDict["basis"] = clean_string(ReadLine[1].split()[0])
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
    try:
        InfoDict["smiles"] = xyz2smi("coords.xyz")
    except FileNotFoundError:
        print("OBabel executable not found in PATH.")
    return InfoDict


def split_zmat(zmat_list):
    """
        Function that will convert a ZMAT into a dictionary.
    """
    connectivity = list()
    parameters = list()
    # Loop over list containing ZMAT strings; first line is
    # comment line
    read_next = False
    for line in zmat_list[1:]:
        if line == "\n":
            if read_next is True:
                break
            else:
                read_next = True
        if read_next is False:
            connectivity.append(line)
        if read_next is True:
            parameters.append(line)
    return connectivity, parameters

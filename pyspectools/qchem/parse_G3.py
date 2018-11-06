"""
parseG3.py

A function to read in the G3 output of a Gaussian calculation, and format
it into a operatable dictionary. The rotational constants and frequencies
are also read in for state counts.
"""

import numpy as np

def parseG3(filepath):
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

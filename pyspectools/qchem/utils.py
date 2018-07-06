
"""
    Contains all of the auxillary functions for CFOURviewer; i.e. file I/O,
    storage to HDF5, copying and pasting, etc.

    The scope of HDF5 will be to store the parsed data, as well as the full
    output file as a string.

    Settings will be stored in a dot folder in the user's home directory;
    this includes templates for the PBS script as well as CFOURviewer settings.
"""

import os
import shutil
import datetime
import yaml
from subprocess import Popen, PIPE
from glob import glob


def generate_folder():
    # Function to generate the next folder in the ID chain.
    # Returns the next ID number in the chain as an integer.
    settings = read_settings()
    dir_list = glob(settings["calc_dir"] + "/*")
    filtered = list()
    for dir in dir_list:
        # This takes only folder names that are numeric
        try:
            filtered.append(int(dir))
        except TypeError:
            pass
    next_ID = max(filtered) + 1
    os.mkdir(settings["calc_dir"] + str(next_ID))
    return next_ID


"""
    File I/O

    Includes YAML and HDF5 functions

    HDF5 system is organized into IDs - an ID can contain one or several
    calculations, and the attributes of an ID group are metadata regarding
    the calculation batch, i.e. a SMILES code to identify the molecule.

    Each calculation is then stored as datasets within this group, and the
    parsed results of the calculation.
"""

def write_yaml(yaml_path, contents):
    # Function for writing dictionary to YAML file
    with open(yaml_path, "w+") as write_file:
        yaml.dump(contents, write_file, default_flow_style=False)


def read_yaml(yaml_path):
    # Function for reading in a YAML file
    with open(yaml_path) as read_file:
        return yaml.load(read_file)


def xyz2smi(filepath):
    # Calls external obabel executable to convert an input xyz file
    # Returns a SMILES string for identification
    proc = Popen(
        ["obabel", "-ixyz", filepath, "-osmi"],
        stdout=PIPE
    )
    output = proc.communicate()[0].decode()
    smi = output.split("\t")[0]
    return smi


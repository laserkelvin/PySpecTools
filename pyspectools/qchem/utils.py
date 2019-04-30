
"""
    Contains all of the auxillary functions for CFOURviewer; i.e. file I/O,
    storage to HDF5, copying and pasting, etc.

    The scope of HDF5 will be to store the parsed data, as well as the full
    output file as a string.

    Settings will be stored in a dot folder in the user's home directory;
    this includes templates for the PBS script as well as CFOURviewer settings.
"""

from subprocess import Popen, PIPE, run


"""
    File I/O

    Includes YAML and HDF5 functions

    HDF5 system is organized into IDs - an ID can contain one or several
    calculations, and the attributes of an ID group are metadata regarding
    the calculation batch, i.e. a SMILES code to identify the molecule.

    Each calculation is then stored as datasets within this group, and the
    parsed results of the calculation.
"""


def obabel_smi(filepath, format="xyz"):
    # Calls external obabel executable to convert an input xyz file
    # Returns a SMILES string for identification
    proc = run(
        ["obabel", "-i{}".format(format), filepath, "-osmi"],
        capture_output=True
    )
    output = proc.stdout.decode()
    smi = output.split("\t")[0]
    return smi


def obabel_png(filepath, format="xyz"):
    """
    Call open-babel to convert a obabel readable format file into a PNG. The conversion requires
    that obabel linked to libcairo, and will dump the image in the current working directory.

    Parameters
    ----------
    filepath: str
        Filepath to the target file
    """
    filename = filepath.split("/")[-1].split(".")[0]
    with open("{}.png".format(filename), "w+") as out_file:
        proc = run(
            ["obabel", "-i{}".format(format), filepath, "-opng"],
            stdout=out_file
        )


def clean_string(string):
    symbols = ['(', ')', '"', '\n']
    for symbol in symbols:
        string = string.replace(symbol, '')
    return string


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

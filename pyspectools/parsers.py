
import os
import struct
from glob import glob

import pandas as pd
import numpy as np

def parse_spectrum(filename, threshold=20.):
    """ Function to read in a blackchirp or QtFTM spectrum from file """
    dataframe = pd.read_csv(
        filename, delimiter="\t", names=["Frequency", "Intensity"], skiprows=1
    )
    dataframe.dropna(inplace=True)
    return dataframe[dataframe["Intensity"] <= threshold]


def parse_ascii(filename, delimiter="\t", names=None, header=None, skiprows=0):
    """
    Generic ASCII parser wrapping the pandas read_csv function.
    Parameters
    ----------
    filename
    delimiter
    names
    header
    skiprows

    Returns
    -------

    """
    dataframe = pd.read_csv(
        filename, delimiter=delimiter, names=names, header=header, skiprows=skiprows
    )
    dataframe.dropna(inplace=True)
    return dataframe


def parse_lin(filename):
    """
        Function to read in a line file, formatted in the SPFIT
        convention.
    """
    data = list()
    with open(filename) as read_file:
        for line in read_file:
            line_data = list()
            # Get all the delimiting out
            split_line = line.split()
            split_cols = split_line[-3:]
            # Convert frequency, uncertainty, and weight
            # into floats
            for col in split_cols:
                try:
                    line_data.append(
                        float(col)
                        )
                except ValueError:
                    line_data.append(0.)
            # Split up the quantum numbers
            #qnos = qnos.split()
            #qnos = [int(num) for num in qnos]
            line_data.append(",".join(split_line[:-3]))
            data.append(line_data)
    dataframe = pd.DataFrame(
        data=data,
        columns=["Frequency", "Uncertainty", "Weight", "Quantum numbers"]
        )
    return dataframe


def parse_cat(simulation_path, low_freq=0., high_freq=np.inf, threshold=-np.inf):
    """
    Parses a simulation output, and filters the frequency and intensity to give
    a specific set of lines.

    The only argument that is required is the path to the simulation output. Others
    are optional, and will default to effectively not filter.

    The quantum numbers are read in assuming hyperfine structure, and thus
    might not be accurate descriptions of what they actually are.
    """
    cat_df = pd.read_fwf(
        simulation_path,
        widths=[13,8,8,2,10,3,7,4,2,2,2,8,2,2],
        header=None
    )
    cat_df.columns = [
        "Frequency",
        "Uncertainty",
        "Intensity",
        "DoF",
        "Lower state energy",
        "Degeneracy",
        "ID",
        "Coding",
        "N'",
        "F'",
        "J'",
        "N''",
        "F''",
        "J''",
    ]
    cat_df = cat_df.loc[
        (cat_df["Frequency"].astype(float) >= low_freq) &  # threshold the simulation output
        (cat_df["Frequency"].astype(float) <= high_freq) &  # based on user specified values
        (cat_df["Intensity"].astype(float) >= threshold)          # or lack thereof
        ]
    return cat_df


def parse_blackchirp(dir_path):
    """
    Function for reading in a Blackchirp experiment. The required input should point to the directory
    containing the Blackchirp files with the correct extensions: .hdr, .tdt, and .fid

    Parameters
    ----------
    dir_path - str
        Filepath pointing to the directory containing the Blackchirp experiment files.

    """
    # read in header information
    hdr_file = glob(os.path.join(dir_path, "*.hdr"))
    header = dict()
    try:
        hdr_file = hdr_file[0]
    except IndexError:
        raise Exception("Header file is missing!")
    with open(hdr_file) as hdr:
        for line in hdr:
            if not line:
                continue
            l = line.split("\t")
            if not l or len(l) < 3:
                continue

            key = l[0].strip()
            value = l[1].strip()
            unit = l[2].strip()

            header[key] = {"value": value, "unit": unit}

    fid_files = glob(os.path.join(dir_path, "*.fid"))
    if len(fid_files) < 1:
        raise Exception("No FID files present!")
    else:
        for file in fid_files:
            with open(file, "rb") as fidfile:
                buffer = fidfile.read(4)
                ms_len = struct.unpack(">I", buffer)
                buffer = fidfile.read(ms_len[0])
                magic_string = buffer.decode('ascii')
                if not magic_string.startswith("BCFID"):
                    raise ValueError("Could not read magic string from {}".format(fidfile.name))

                l = magic_string.split("v")
                if len(l) < 2:
                    raise ValueError("Could not determine version number from magic string {}".format(magic_string))

                version = l[1]

                buffer = fidfile.read(4)
                fidlist_size = struct.unpack(">I", buffer)[0]
                for i in range(0, fidlist_size):
                    pass
            self.fid_list.append(BlackChirpFid(version, fidfile))

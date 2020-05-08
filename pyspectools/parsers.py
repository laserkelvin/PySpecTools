import os
import struct
from glob import glob

import pandas as pd
import numpy as np

from pyspectools import ftmw_analysis as fa


def parse_spectrum(filename, threshold=20.0):
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
                    line_data.append(float(col))
                except ValueError:
                    line_data.append(0.0)
            # Split up the quantum numbers
            # qnos = qnos.split()
            # qnos = [int(num) for num in qnos]
            line_data.append(",".join(split_line[:-3]))
            data.append(line_data)
    dataframe = pd.DataFrame(
        data=data, columns=["Frequency", "Uncertainty", "Weight", "Quantum numbers"]
    )
    return dataframe


def parse_cat(simulation_path, low_freq=0.0, high_freq=np.inf, threshold=-np.inf):
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
        widths=[13, 8, 8, 2, 10, 3, 7, 4, 2, 2, 2, 8, 2, 2],
        header=None,
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
        (cat_df["Frequency"].astype(float) >= low_freq)
        & (  # threshold the simulation output
            cat_df["Frequency"].astype(float) <= high_freq
        )
        & (  # based on user specified values
            cat_df["Intensity"].astype(float) >= threshold
        )  # or lack thereof
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
    # Read in header information
    hdr_file = glob(os.path.join(dir_path, "*.hdr"))
    header = dict()
    try:
        hdr_file = hdr_file[0]
        exp_id = hdr_file.split("/")[-1].split(".")[0]
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

    # Locate all the FIDs
    fid_files = glob(os.path.join(dir_path, "*.fid"))
    if len(fid_files) < 1:
        raise Exception("No FID files present!")
    else:
        fid_list = list()
        for file in fid_files:
            with open(file, "rb") as fidfile:
                buffer = fidfile.read(4)
                ms_len = struct.unpack(">I", buffer)
                buffer = fidfile.read(ms_len[0])
                magic_string = buffer.decode("ascii")
                if not magic_string.startswith("BCFID"):
                    raise ValueError(
                        "Could not read magic string from {}".format(fidfile.name)
                    )

                l = magic_string.split("v")
                if len(l) < 2:
                    raise ValueError(
                        "Could not determine version number from magic string {}".format(
                            magic_string
                        )
                    )

                version = l[1]

                buffer = fidfile.read(4)
                fidlist_size = struct.unpack(">I", buffer)[0]
                for i in range(0, fidlist_size):
                    # Create a BlackChirpFid object
                    fid_list.append(fa.BlackChirpFid.from_binary(fidfile))

    time_data = dict()
    tdt_file = glob(os.path.join(dir_path, "*.tdt"))
    try:
        tdt_file = tdt_file[0]
    except IndexError:
        raise Exception("Time stamp data is missing!")
    with open(tdt_file) as tdt:
        look_for_header = True
        header_list = []
        for line in tdt:
            print(line)
            if line.strip() == "":
                continue
            if line.startswith("#") and "PlotData" in line:
                look_for_header = True
                header_list = []
                continue
            if line.startswith("#"):
                continue

            l = line.split("\t")
            if len(l) < 1:
                continue

            if look_for_header is True:
                for i in range(0, len(l)):
                    name = ""
                    l2 = str(l[i]).split("_")
                    for j in range(0, len(l2) - 1):
                        name += str(l2[j]).strip()
                    time_data[name] = []
                    header_list.append(name)
                look_for_header = False
            else:
                for i in range(0, len(l)):
                    time_data[header_list[i]].append(str(l[i]).strip())
    return exp_id, header, fid_list, time_data


def read_binary_fid(filepath):
    """
    Read in a binary Blackchirp FID file. This is based on the original code by Kyle Crabtree, with some minor
    perfomance improvements by Kelvin Lee. The only difference is most of the for loops for reading the points
    have been replaced by numpy broadcasts.

    Parameters
    ----------
    filepath - str
        Filepath to the Blackchirp .fid file

    Returns
    -------
    param_dict - dict
        Contains header information about the FID, such as the number of shots, point spacing, etc.
    xy_data - 2-tuple of numpy 1D array
        Contains two columns; xy_data[0] is the time data in microseconds, and xy_data[1] is the
        signal.
    raw_data - numpy 1D array
        Contains the raw, uncorrected ADC sums. The signal data is converted from this by scaling
        it with the multiplication factor v_mult.

    """
    with open(filepath) as read_file:
        read_str = ">3dqHbI"
        d = struct.unpack(read_str, read_file.read(struct.calcsize(read_str)))
        spacing = d[0] * 1e6
        probe_freq = d[1]
        v_mult = d[2]
        shots = d[3]
        if d[4] == 1:
            sideband = -1.0
        else:
            sideband = 1.0
        point_size = d[5]
        size = d[6]

        param_dict = {
            "spacing": spacing,
            "probe_freq": probe_freq,
            "v_mult": v_mult,
            "shots": shots,
            "point_size": point_size,
            "size": size,
            "sideband": sideband,
        }

        if point_size == 2:
            read_string = ">" + str(size) + "h"
            dat = struct.unpack(
                read_string, read_file.read(struct.calcsize(read_string))
            )
        elif point_size == 3:
            for i in range(0, size):
                chunk = read_file.read(3)
                dat = struct.unpack(
                    ">i", (b"\0" if chunk[0] < 128 else b"\xff") + chunk
                )[0]
        elif point_size == 4:
            read_string = ">" + str(size) + "i"
            dat = struct.unpack(
                read_string, read_file.read(struct.calcsize(read_string))
            )
        elif point_size == 8:
            read_string = ">" + str(size) + "q"
            dat = struct.unpack(
                read_string, read_file.read(struct.calcsize(read_string))
            )
        else:
            raise ValueError("Invalid point size: " + str(point_size))
        # Now read in the data with broadcasting
        raw_data = np.array(dat[:size])
        data = raw_data * v_mult / shots
        x_data = np.linspace(0.0, size * spacing, int(size))
        xy_data = np.vstack((x_data, data))
    return param_dict, xy_data, raw_data


def parse_fit(filepath):
    """
    Function to parse the output of an SPFIT .fit file. This version of the code is barebones compared to the
    previous iteration, which provides more feedback. This version simply returns a dictionary containing the
    obs - calc for each line, the fitted parameters, and the microwave RMS.

    Parameters
    ----------
    filepath: str
        Filepath to the .fit file to parse.

    Returns
    -------
    fit_dict: dict
        Dictionary containing the parsed data.
    """
    fit_dict = {"o-c": {}, "parameters": {}, "rms": None}
    with open(filepath) as read_file:
        lines = read_file.readlines()
    for index, line in enumerate(lines):
        # Read the obs - calc on individual lines
        if "EXP.FREQ." in line:
            stop_flag = False
            entry_index = 1
            line_dict = dict()
            while stop_flag is False:
                entry = lines[index + entry_index].split()
                if entry[0] == "NORMALIZED" or entry[0] == "Fit":
                    stop_flag = True
                elif entry[1] == "NEXT" or entry[1] == "Lines":
                    entry_index += 1
                    pass
                else:
                    # Read in the line information
                    line_dict[entry_index] = {
                        "o-c": float(entry[-3]),
                        "qnos": entry[1:-5],
                        "frequency": entry[-5],
                    }
                    entry_index += 1
        if "NEW PARAMETER" in line:
            stop_flag = False
            entry_index = 1
            param_dict = dict()
            while stop_flag is False:
                entry = lines[index + entry_index]
                for bracket in ["""(""", """)"""]:
                    entry = entry.replace(bracket, " ")
                entry = entry.split()
                if entry[0] != "MICROWAVE":
                    coding = int(entry[1])
                    param_dict[coding] = float(entry[-3])
                    entry_index += 1
                else:
                    stop_flag = True
        if "MICROWAVE RMS" in line:
            fit_dict["microwave_rms"] = float(line.split()[3])
        if "NEW RMS ERROR" in line:
            fit_dict["rms"] = float(line.split()[-2])
    fit_dict["o-c"] = line_dict
    fit_dict["parameters"] = param_dict
    return fit_dict


import pandas as pd


def parse_spectrum(filename, threshold=20.):
    """ Function to read in a blackchirp or QtFTM spectrum from file """
    dataframe = pd.read_csv(
        filename, delimiter="\t", names=["Frequency", "Intensity"], skiprows=1
    )
    return dataframe[dataframe["Intensity"] <= threshold]


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

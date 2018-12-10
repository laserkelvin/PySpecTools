
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
            qnos = line[:12]
            split_cols = line[13:].split()
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
            qnos = qnos.split()
            qnos = [int(num) for num in qnos]
            line_data.append(qnos)
            data.append(line_data)
    dataframe = pd.DataFrame(
        data=data,
        columns=["Frequency", "Uncertainty", "Weight", "Quantum numbers"]
        )
    return dataframe


""" Routines for fitting double resonance data """

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import peakutils
from pyspectools import mpl_settings
from matplotlib import pyplot as plt

def parse_data(filepath):
    """ Function to read in a DR file. When there are multiple columns present
        in the file, this is interpreted as multiple channels being used to
        monitor during the DR experiment. In this case, we do the analysis on
        the co-average of these columns.
    """
    df = pd.read_csv(filepath, delimiter="\t", index_col=0, header=None, skiprows=1)
    if len(df.keys()) > 1:
        # co-average spectra if there are more than one columns
        df["average"] = np.average([df[column].values for column in list(df.keys())], axis=0)
    return df

def clean_data(dataframe, column=1, baseline=False, freqrange=[0., np.inf]):
    """ Routine for pre-processing the DR fits.
        1. Option to remove baseline
        2. Option to truncate the detection range
    """
    # If true, we perform baseline subtraction
    if baseline is True:
        bline = peakutils.baseline(dataframe[column].astype(float), deg=2)
        dataframe["baseline subtracted"] = dataframe[column].astype(float) - bline
    # Find nearest indices for values
    lower_index = dataframe.index.searchsorted(freqrange[0])
    upper_index = dataframe.index.searchsorted(freqrange[1])
    # Return the truncated dataframe
    return dataframe.loc[dataframe.index[lower_index:upper_index]]

def gaussian(x, A, c, w, offset):
    # stock Gaussian distribution. The sign of A is such that the function
    # is flipped upside down.
    return -A * np.exp(-(c - x)**2. / 2. * w**2.) + offset

def plot_data(dataframe, frequency=None):
    """ Function to plot the DR data, and the fitted Gaussian peak. """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Intensity")
    ax.set_ylim([dataframe.min().min() - 0.03, dataframe.max().max()]) # Set the ylimits

    if "average" in list(dataframe.keys()):
    # If the DR has co-averaging, we'll plot that too
        for column in [columns for columns in list(dataframe.keys()) if columns not in ["average", "baseline subtracted", "fit"]]:
            ax.plot(dataframe.index, dataframe[column], label="channel " + str(column), alpha=0.3)
        ax.plot(dataframe.index, dataframe["average"], label="Co-averaged")
    if "baseline subtracted" in list(dataframe.keys()):
    # If the DR has a baseline subtracted, plot the outcome of that too
        ax.plot(dataframe.index, dataframe["baseline subtracted"], label="Baseline subtracted")
    if "fit" in list(dataframe.keys()):
    # Plot the fit result
        ax.plot(dataframe.index, dataframe["fit"], label="Fit", lw=2.)
    if frequency is not None:
    # If we managed to find a center frequency, fit a straight line through
    # and annotate the graph with the peak frequency
        ax.text(frequency + 0.1, dataframe.min().min() - 0.02, "Center: %5.3f" % frequency, size="x-large")
        ax.vlines(frequency, *ax.get_ylim())

    ax.legend()
    plt.tight_layout()
    return fig, ax

def fit_dr(dataframe, column=1, bounds=None):
    # Determine an estimate for the peak depletion

    peak_guess = dataframe[column].idxmin()
    print("Guess for center frequency: " + str(peak_guess))

    if bounds is None:
        bounds = ([0., peak_guess - 0.1, 1., 0.,],
                  [np.inf, peak_guess + 0.1, 20., np.inf]
                 )

    try:
        optimized, covariance = curve_fit(
            gaussian,
            dataframe.index,
            dataframe[column].astype(float),
            p0=[1., peak_guess, 1.0, 0.0],
            bounds=bounds
        )
    except RuntimeError:
        print("No optimal solution found. Try narrowing the frequency range of the data.")
        optimized = [None] * 4
    print(optimized)
    dataframe["fit"] = gaussian(dataframe.index, *optimized)
    return dataframe, optimized

def analyze_dr(filepath, baseline=False, freqrange=[0., np.inf]):
    """ Main driver function that will perform all of the operations for
        analyzing a DR spectrum.

        What will normally need to be fiddled with is the frequency range to
        analyze. Preferably, we chop off parts of the spectrum that will inter-
        fere with the detection of the depletion maximum, but enough to get
        a sense of dynamic range in the spectrum.
    """
    filename = os.path.splitext(filepath)[0]

    dataframe = parse_data(filepath)
    if "average" in list(dataframe.keys()):
        column = "average"      # if we co-averaged, work on that
    else:
        column = 1              # Default to first column
    dataframe = clean_data(dataframe, column, baseline=baseline, freqrange=freqrange)

    if baseline is True and column == "average":
        column = "baseline subtracted"
    elif baseline is False and column == "average":
        column = "average"
    else:
        column = 1
    print("Using column " + str(column))
    dataframe, optimized = fit_dr(dataframe, column)

    fig, ax = plot_data(dataframe, frequency=optimized[1])

    ax.set_title(filename)

    dataframe.to_csv(filename + "_fit.csv")
    fig.savefig(filename + "_fit.pdf", format="pdf")

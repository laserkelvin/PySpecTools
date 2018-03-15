
""" Routines for fitting double resonance data """

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import peakutils
from uncertainties import ufloat
from matplotlib import pyplot as plt

def parse_data(filepath):
    """ Function to read in a DR file. When there are multiple columns present
        in the file, this is interpreted as multiple channels being used to
        monitor during the DR experiment. In this case, we do the analysis on
        the co-average of these columns.
    """
    df = pd.read_csv(filepath, delimiter="\t", index_col=0, header=None, skiprows=1, comment="#")
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

def plot_data(dataframe, fitresults=None):
    """ Function to plot the DR data, and the fitted Gaussian peak. """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    fig.subplots_adjust(top=0.55)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Intensity")
    ax.set_ylim([dataframe.min().min() - 0.03, dataframe.max().max()]) # Set the ylimits
    min_freq = dataframe.index.min()
    max_freq = dataframe.index.max()
    ax.set_xticks(np.arange(min_freq, max_freq, (max_freq - min_freq) / 4.))

    if "average" not in list(dataframe.keys()):
        ax.plot(dataframe.index, dataframe[1], label="Data")
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
        if fitresults is not None:
        # If we managed to find a center frequency, fit a straight line through
        # and annotate the graph with the peak frequency
            text_to_write = ""
            for index, name in enumerate(["Amplitude", "Center", "Width", "Offset"]):
                text_to_write+= name + ": " + "{:.3uS}".format(fitresults[index]) + "\n"
                #if type(frequency) == type(ufloat(1., 2.)):
                #    freq = frequency.n
                #    form_freq = "{:.3uS}".format(frequency)
                #ax.text(freq + 0.1, dataframe.min().min() - 0.02, "Center: " + form_freq, size="x-large")
            ax.text(0.8, 0.2, text_to_write, size="x-large", ha="center", va="center", transform=ax.transAxes)
            ax.vlines(fitresults[1].n, *ax.get_ylim())
            if fitresults[1].s <= 0.5:
                ax.fill_between([fitresults[1].n - fitresults[1].s, fitresults[1].n + fitresults[1].s],
                                y1=dataframe["fit"].min(),
                                y2=dataframe["fit"].max(),
                                facecolor="#2b8cbe",
                                alpha=0.5
                                )

    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def fit_dr(dataframe, column=1, bounds=None):
    # Determine an estimate for the peak depletion; uses the minimum value in
    # intensity as an initial guess for the fit.

    peak_guess = dataframe[column].idxmin()
    print("Guess for center frequency: " + str(peak_guess))

    if bounds is None:
        bounds = ([0., peak_guess - 1., 0.2, 0.,],
                  [np.inf, peak_guess + 1., 10., np.inf]
                 )

    try:
        optimized, covariance = curve_fit(
            gaussian,
            dataframe.index,
            dataframe[column].astype(float),
            p0=[1., peak_guess, 0.5, 0.0],
            bounds=bounds
        )
    except RuntimeError:
        print("No optimal solution found. Try narrowing the frequency range of the data.")
        optimized = [None] * 4
    dataframe["fit"] = gaussian(dataframe.index, *optimized)
    return dataframe, optimized, covariance

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
    # Perform the curve fitting
    dataframe, optimized, covariance = fit_dr(dataframe, column)
    # Calculate the standard deviation; this is based on the diagonal elements
    # of the covariance matrix. If the off-diagonal elements are large, there
    # is significant correlation between variables and the answers will be
    # quite uncertain...
    stdev = np.sqrt(np.diag(covariance))
    results = list()

    # Print the results - short-hand notation is used
    print("Final fitting results for " + filename)
    for index, name in enumerate(["Amplitude", "Center", "Width", "Offset"]):
        result = ufloat(optimized[index], stdev[index])
        print(name + ":    " + "{:.3uS}".format(result))
        results.append(result)

    # Plot the resulting DR fit for physical printing
    fig, ax = plot_data(dataframe, fitresults=results)

    ax.set_title(filename)

    dataframe.to_csv(filename + "_fit.csv")
    fig.savefig(filename + "_fit.pdf", format="pdf", dpi=300, bbox_inches='tight')

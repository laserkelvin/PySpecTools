
from astroquery.splatalogue import Splatalogue
from astropy import units as u
from lmfit import models
import numpy as np
import pandas as pd
import peakutils
from . import plotting


def fit_line_profile(spec_df, frequency, intensity=None, name=None, verbose=False):
    """ Low level function wrapper to to fit line profiles
        in chirp spectra.
    """
    model = models.VoigtModel()
    params = model.make_params()
    # Set up boundary conditions for the fit
    params["center"].set(
        frequency,
        min=frequency - 0.05,
        max=frequency + 0.05
    )
    # If an intensity is supplied
    if intensity:
        params["amplitude"].set(intensity)
    params["sigma"].set(
        0.05,
        min=0.04,
        max=0.07
    )
    freq_range = [frequency + offset for offset in [-5., 5.]]
    slice_df = spec_df[
        (spec_df["Frequency"] >= freq_range[0]) & (spec_df["Frequency"] <= freq_range[1])
    ]
    # Fit the peak lineshape
    fit_results = model.fit(
        slice_df["Intensity"],
        params,
        x=slice_df["Frequency"],
    )
    if verbose is True:
        print(fit_results.fit_report())
    yfit = fit_results.eval(x=spec_df["Frequency"])
    if name:
        spec_df[name] = yfit
    # Subtract the peak contribution
    spec_df["Cleaned"] -= yfit


def peak_find(spec_df, col="Intensity", thres=0.015):
    """ Wrapper for peakutils applied to pandas dataframes """
    peak_indices = peakutils.indexes(
        spec_df[col],
        thres=thres
        )
    return spec_df.iloc[peak_indices]


def search_center_frequency(frequency, width=0.5):
    """ Wrapper for the astroquery Splatalogue search
        This function will take a center frequency, and query splatalogue
        within the CDMS and JPL linelists for carriers of the line.

        Input arguments:
        frequency - float specifying the center frequency
    """
    min_freq = frequency - width
    max_freq = frequency + width
    splat_df = Splatalogue.query_lines(
        min_freq*u.MHz,
        max_freq*u.MHz,
        line_lists=["CDMS", "JPL"]
    ).to_pandas()
    # These are the columns wanted
    columns = [
        "Species",
        "Chemical Name",
        "Meas Freq-GHz",
        "Freq-GHz",
        "Resolved QNs",
        "CDMS/JPL Intensity"
        ]
    return splat_df[columns]


def assign_peaks(spec_df, frequencies, **kwargs):
    """ Higher level function that will help assign features
        in a chirp spectrum.

        Input arguments:
        spec_df - pandas dataframe containing chirp data
        frequencies - iterable containing center frequencies of peaks
        Optional arguments are passed into the peak detection as well
        as in the plotting functions.

        Returns a dataframe containing all the assignments, and a list
        of unidentified peak frequencies.
    """
    unassigned = list()
    dataframes = pd.DataFrame()
    for freq_index, frequency in enumerate(frequencies):
        splat_df = search_center_frequency(frequency)
        nitems = len(splat_df)
        # Only act if there's something found
        if nitems > 0:
            splat_df["Dev. Measured"] = np.abs(
                (splat_df["Meas Freq-GHz"] * 1000.) - frequency
                )
            splat_df["Dev. Calc"] = np.abs(
                (splat_df["Freq-GHz"] * 1000.) - frequency
                )
            print(splat_df)
            try:
                print("Peak frequency is " + str(frequency))
                index = int(
                    input(
                        "Please choose an assignment index: 0-" + str(nitems - 1)
                        )
                )
                assigned = True
            except ValueError:
                print("Deferring assignment")
                unassigned.append(frequency)
                assigned = False
            # Take the assignment. Double brackets because otherwise
            # a series is returned rather than a dataframe
            if assigned is True:
                assignment = splat_df.iloc[[index]].sort_values(
                    ["Dev. Measured", "Dev. Calc"],
                    ascending=True
                )
                ass_freq = assignment["Meas Freq-GHz"]
                # If the measurement is not available, go for
                # the predicted value
                if ass_freq is np.nan:
                    ass_freq = assignment["Freq-GHz"]
                ass_name = assignment["Species"] + "-" + assignment["Resolved QNs"]
                # Clean the line and assign it
                #fit_line_profile(spec_df, frequency, name=ass_name)
                # Keep track of the assignments in a dataframe
                dataframes.append(assignment)
            print("----------------------------------------------------")
        else:
            print("No species known for " + str(frequency))
    return dataframes, unassigned

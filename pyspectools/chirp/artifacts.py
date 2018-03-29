
import numpy as np
import peakutils
import pandas as pd
from lmfit import models
from . import parsers


def artifact_detection(spec_path, freq_range=[8000., 19000.], **kwargs):
    """ Function for detecting and storing chirp artifacts
        for later use.

        The input parameters are:
        spec_path - path to a chirp spectrum
        freq_range - frequency range

        Additional kwargs can be passed to the peak
        detection functions; this wraps around the
        peakutils functions.
    """
    # Remove any directory and extensions from the name
    filename = spec_path.split("/")[-1].split(".")[0]
    spec_df = parsers.parse_spectrum(spec_path)
    min_freq = min(freq_range)
    max_freq = max(freq_range)
    # Filter frequency range considered
    spec_df = spec_df[(spec_df["Frequency"] >= min_freq) & (spec_df["Frequency"] <= max_freq)]
    # Put in a default value for peak detection - this works
    # relatively well for normal artifact detection
    if "thres" not in kwargs:
        kwargs["thres"] = 0.01
    peak_indices = peakutils.indexes(
        spec_df["Intensity"],
        **kwargs
    )
    # Slice dataframe to only include peaks
    artifact_df = spec_df.iloc[peak_indices]
    artifact_df.to_csv(filename + ".artifacts.csv", sep="\t", index=False)
    return artifact_df


def remove_artifacts(spec_df, artifact_path, verbose=False):
    """ Function for removing artifacts from a spectrum.
        A csv containing artifacts is loaded, and goes through
        fitting and subtracting them from the actual spectrum.

        Input arguments are:
        spec_df - dataframe containing the actual spectrum you want
        to clean
        artifact_path - path to a csv file containing the frequency
        and peak intensities
        verbose - if True, prints out the fitting results
    """
    artifact_df = parsers.parse_spectrum(artifact_path)
    spec_df["Cleaned"] = spec_df["Intensity"].values
    # Loop over all of the peaks
    for index, row in artifact_df.iterrows():
        # Designate a Lorentzian lineshape for the peaks
        model = models.VoigtModel()
        params = model.make_params()
        # Set up boundary conditions for the fit
        params["center"].set(
            row["Frequency"],
            min=row["Frequency"] - 0.05,
            max=row["Frequency"] + 0.05
        )
        params["amplitude"].set(row["Intensity"])
        params["sigma"].set(
            0.05,
            min=0.04,
            max=0.06
        )
        freq_range = [row["Frequency"] + offset for offset in [-5., 5.]]
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
        # Subtract the peak contribution
        spec_df["Cleaned"] -= fit_results.eval(x=spec_df["Frequency"])

""" Routines for fitting double resonance data """

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from lmfit import Model
from lmfit.models import GaussianModel, LinearModel


def parse_data(filepath):
    """ Function to read in a DR file. When there are multiple columns present
        in the file, this is interpreted as multiple channels being used to
        monitor during the DR experiment. In this case, we do the analysis on
        the co-average of these columns.
    """
    # Skip reading the header - we're doing it ourselves
    df = pd.read_csv(filepath, sep="\t", header=None, skiprows=1, comment="#")
    # Columns denoted by number
    cols = list(np.arange(1, len(df.columns)))
    full_cols = ["Frequency"]
    full_cols.extend(cols)
    # Rename to generalize
    df.columns = full_cols
    stats = list()
    # Set up vector for performing averages
    average = np.zeros(len(df["Frequency"]))
    for index, column in enumerate(df):
        if "Frequency" not in str(column) and "BS" not in str(column):
            df[str(column) + "-N"] = df[column]
            # Make sure the baseline is the same for all of the
            # spectra - subtract off the "noise average"
            # and offset by 1 to make it easily convertible into
            # a percentage
            df[str(column) + "-BS"] -= df[column].mean()
            df[str(column) + "-BS"] += 1.0
            # for weighting the averages
            weight = df[column].std()
            # Calculate the weighted average
            average += (df[column] * weight) / weight
    # Create composite average - this may or may not be used
    average /= index + 1
    average -= np.mean(average)
    average += 1
    df["Average-BS"] = average
    df["Average"] = np.average(df[cols], axis=1)
    return df


def init_dr_model(baseline=False, guess=None):
    """ Function to serialize a Model object for DR fits.
        Defaults to a Gaussian line shape, but the option to include
        a linear offset is available.

        Additionally, if a guess is provided in the
        lmfit convention then it will be incorporated.

        args:
        -----------------
        baseline - boolean indicating if linear offset is included
        guess - nested dict with keys corresponding to parameter name
                and values corresponding to dicts with key/value
                indicating the parameter value, range, and constraints
    """
    model = GaussianModel()
    if baseline:
        model += LinearModel()
    params = model.make_params()
    if guess:
        params.update(guess)
    # Constrain amplitude to always be negative
    # since we're measuring depletion.
    params["amplitude"].set(max=0.0)
    return model, params


def peak_detect(dataframe, col="Average", interpolate=True):
    """ Routine for peak detection to provide an initial
        guess for the center frequency.

        The algorithm assumes that the ten smallest intensities
        corresponds to where the depletion approximately is.

        By default, a cubic spline is performed on the intensities
        so that a finer grid of points gives a better determination
        of the center frequency.
    """
    if interpolate:
        interpolant = interp1d(dataframe["Frequency"], dataframe[col], "cubic")
        # Interpolate between the min and max with 10 times more
        # points than measured
        new_x = np.linspace(
            dataframe["Frequency"].min(),
            dataframe["Frequency"].max(),
            len(dataframe["Frequency"]) * 20,
        )
        new_y = interpolant(new_x)
        # Set up dataframe for detection
        detect_df = pd.DataFrame(
            data=list(zip(new_x, new_y)), columns=["Frequency", col]
        )
    else:
        # Use the experimental data
        detect_df = dataframe
    # Use pandas to find the 10 smallest values
    trunc_df = detect_df.nsmallest(10, [col], "first")
    avg_freq = trunc_df["Frequency"].mean()
    # Guess amplitude given by negative value
    guess_ampl = trunc_df[col].min() - trunc_df[col].mean()
    return avg_freq, guess_ampl


def fit_dr(dataframe, col="Average", baseline=True, guess=None):
    """ Function for fitting double resonance data.
        
        Required input:
        -----------------
        dataframe - pandas dataframe containing

        args:
        -----------------
        col - str denoting which column to use for fitting.
        baseline - boolean denoting whether a linear baseline is fit
        guess - dict for fitting parameters; this is passed into
                the fit directly and so must match the lmfit convention.
    """
    model, params = init_dr_model(baseline, guess)
    if guess is None:
        center, guess_ampl = peak_detect(dataframe, col)
        # Center guess is constrained to 500 kHz
        params["center"].set(center, min=center - 0.5, max=center + 0.5)
        params["amplitude"].set(guess_ampl, max=0.0, min=-10.0)
        params["sigma"].set(min=0.0, max=10.0)
        if baseline:
            pass
            # params["intercept"].set(min=-., max=5.)
            # params["slope"].set(min=-5., max=5.)

    result = model.fit(dataframe[col], x=dataframe["Frequency"], params=params)

    dataframe["Fit"] = result.best_fit
    dataframe.fit_result = result
    # Do some extra stuff like figure out the "baseline"
    # for plotting; expressing it in terms of percent
    # depletion
    zero = dataframe["Fit"].max()
    dataframe["Base Signal"] = 100 * (dataframe[col].values / zero)
    dataframe["Base Fit"] = 100 * (dataframe["Fit"].values / zero)
    dataframe["Offset Frequency"] = (
        dataframe["Frequency"].values - result.best_values["center"]
    )
    print(result.fit_report())

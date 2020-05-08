
from typing import List
from pathlib import Path

import numpy as np
import lmfit
import pandas as pd
import os
import peakutils
from matplotlib import pyplot as plt

from . import fft_routines
from . import interpolation
from ..lineshapes import sec_deriv_lorentzian

"""
    Miscellanous routines for parsing and batch processing
    the millimeter-wave data.
"""

def parse_data(filepath):
    """
        Function for parsing data from a millimeter-wave
        data file.
    """
    settings = dict()
    intensity = list()
    # Boolean flags to check when to start/stop
    # reading parameters
    read_params = False
    read_int = False
    read_zeeman = False
    finished = False
    fieldoff_intensities = list()
    fieldon_intensities = list()
    with open(filepath) as read_file:
        for line in read_file:
            if "*****" in line:
                read_int = False
                if finished is True:
                    break
            if "Scan" in line:
                if "[Field ON]" in line:
                    read_zeeman = True
                scan_details = line.split()
                settings["ID"] = int(scan_details[1])
                # settings["Date"] = str(scan_details[4])
                read_params = True
                read_int = False
                continue
            if read_int is True:
                if read_zeeman is False:
                    fieldoff_intensities += [float(value) for value in line.split()]
                else:
                    fieldon_intensities += [float(value) for value in line.split()]
                    finished = True
            if read_params is True and len(line.split()) > 1:
                # Read in the frequency step, frequency, and other info
                # needed to reconstruct the frequency data
                scan_params = line.split()
                shift = 1
                settings["Frequency"] = float(scan_params[0])
                settings["Frequency step"] = float(scan_params[1])
                if len(scan_params) == 4:
                    settings["Multiplier"] = 1.
                    shift = 0
                # If the multiplier data is there, we don't shift the read
                # index over by one
                else:
                    settings["Multiplier"] = float(scan_params[2])
                settings["Center"] = float(scan_params[2 + shift])
                settings["Points"] = int(scan_params[3 + shift])
                read_params = False
                # Start reading intensities immediately afterwards
                read_int = True
                continue
    fieldoff_intensities = np.array(fieldoff_intensities)
    fieldon_intensities = np.array(fieldon_intensities)

    # Generate the frequency grid
    settings["Frequency step"] = settings["Frequency step"] * settings["Multiplier"]
    # This calculates the length of either side
    side_length = settings["Frequency step"] * (settings["Points"] // 2)
    start_freq = settings["Frequency"] - side_length
    end_freq = settings["Frequency"] + side_length
    frequency = np.linspace(start_freq, end_freq, settings["Points"])

    return frequency, fieldoff_intensities, fieldon_intensities, settings


def open_mmw(filepath, **kwargs):
    """
        Function for opening and processing a single millimeter
        wave data file.
        
        Returns a pandas dataframe containing Frequency and
        Intensity information.
    """
    frequency, fieldoff_intensities, fieldon_intensities, settings = parse_data(filepath)
    # Sometimes there are spurious zeros at the end of intensity data;
    # this will trim the padding
    npoints = settings.get("Points")
    fieldoff_intensities = fieldoff_intensities[:npoints]
    if fieldon_intensities.size > 1:
        fieldon_intensities = fieldon_intensities[:npoints]
    # Calculate sample rate in Hz for FFT step
    sample_rate = 1. / settings["Frequency step"] * 1e6
    param_dict = {
        "window_function": None,
        "cutoff": [50, 690],
        "sample_rate": sample_rate
    }
    param_dict.update(**kwargs)

    fieldoff_intensities = fft_routines.fft_filter(fieldoff_intensities, **param_dict)
    # field on is not always there, but if it we use it to calculate Off - On
    if fieldon_intensities.size > 1:
        fieldon_intensities = fft_routines.fft_filter(fieldon_intensities, **param_dict)
        intensity = fieldoff_intensities - fieldon_intensities
    else:
    # if field on is missing, just use zeros instead
        fieldon_intensities = np.zeros(fieldoff_intensities.size)
        intensity = fieldoff_intensities
    # Format the data as a Pandas dataframe
    mmw_df = pd.DataFrame(
        data={"Frequency": frequency, 
              "Field OFF": fieldoff_intensities, 
              "Field ON": fieldon_intensities, 
              "OFF - ON": intensity
              },
    )
    return mmw_df


def test_filtering(path: str, **kwargs):
    """
    Convenience function to open a legacy data file and
    process it; kwargs are passed to the `fft_filter`
    function call. The function then returns a Matplotlib
    figure and axis object, showing the FFT filtered
    spectrum and the time-domain signal chunk that it
    corresponds to.

    Parameters
    ----------
    path : str
        Filepath to the data file.

    Returns
    -------
    fig, ax
        Matplotlib figure/axis objects.
    """
    frequency, off, on, settings = parse_data(path)
    filtered = fft_routines.fft_filter(off, **kwargs)
    
    fig, axarray = plt.subplots(2, 1, figsize=(10, 5))
    
    ax = axarray[0]
    ax.set_title("Frequency domain")
    ax.plot(frequency, filtered)
    ax.set_xlabel("Frequency (MHz)")
    
    ax = axarray[1]
    ax.set_title("Time domain")
    cutoff = kwargs.get("cutoff", np.arange(30, 500))
    ax.plot(np.fft.fft(filtered)[np.arange(*cutoff)])
    
    return fig, ax


def sec_deriv_peak_detection(df_group, threshold=5, window_size=25, magnet_thres=0.5, **kwargs):
    """
    Function designed to take advantage of split-combine-apply techniques to analyze
    concatenated spectra, or a single spectrum. The spectrum is cross-correlated with
    a window corresponding to a second-derivative Lorentzian line shape, with the parameters
    corresponding to the actual downward facing peak such that the cross-correlation
    is upwards facing for a match.
    
    This X-correlation analysis is only done for the Field OFF data; every peak
    should appear in the Field OFF, and should disappear under the presence of
    a magnetic field. If the Field ON spectrum is non-zero, the peak finding is
    followed up by calculating the ratio of the intensity at the same position
    for ON/OFF; if the ratio is below a threshold, we consider it a magnetic line.

    Parameters
    ----------
    df_group : pandas DataFrame
        Dataframe containing the millimeter-wave spectra.
    threshold : int, optional
        Absolute intensity units threshold for peak detection.
        This value corresponds to the value used in the X-correlation
        spectrum, by default 5
    window_size : int, optional
        Size of the second derivative Lorentizan window function, 
        by default 25
    magnet_thres : float, optional
        Threshold for determining if a peak is magnetic, given
        as the ratio of ON/OFF. A value of 1 means the line is
        nominally unaffected by a magnetic field, and less than
        1 corresponds to larger responses to magnetic fields. 
        By default 0.5

    Returns
    -------
    pandas DataFrame
        DataFrame holding the detected peaks and their associated
        magnet tests, if applicable.
    """
    # Get the frequencies and whatnot as numpy arrays; peakutils does not play
    # nicely with Series objects.
    signal = df_group["Field OFF"].to_numpy()
    frequency = df_group["Frequency"].to_numpy()
    # Improve SNR with cross-correlation
    corr_signal = cross_correlate_lorentzian(signal, window_size)
    # Find the peaks that match the specified threshold
    indices = peakutils.indexes(corr_signal, thres=threshold, **kwargs, thres_abs=True)
    peak_subset = df_group.iloc[indices]
    # Only evaluate magnetic if spectrum has a Field ON component
    if peak_subset["Field ON"].sum() != 0.:
        peak_subset.loc[:, "Ratio"] = peak_subset.loc[:,"Field ON"] / peak_subset.loc[:,"Field OFF"]
        peak_subset.loc[:, "Magnetic"] = peak_subset.loc[:,"Ratio"] < magnet_thres
    return peak_subset


def cross_correlate_lorentzian(signal: np.ndarray, window_size=25, **kwargs):
    """
    Calculate the cross-correlation spectrum between an input signal and the
    second derivative lineshape of a Lorentzian function. This is a matched
    filter analysis to extract optimal signal to noise for millimeter-wave
    data.

    Parameters
    ----------
    signal : np.ndarray
        NumPy 1D array containing the raw signal.
    window_size : int
        Size of the window function.

    Returns
    -------
    corr_signal
        NumPy 1D array containing the cross-correlation spectrum.
    """
    params = {
        "x0": 0,
        "gamma": 0.25,
        "I": 1
    }
    if kwargs:
        params.update(**kwargs)
    # Create a template of the second derivative Lorentzian profile for
    # x-correlating with the spectrum
    temp_x = np.linspace(-5, 5, window_size)
    temp_y = sec_deriv_lorentzian(temp_x, **params)
    # Cross-correlate with the Lorentzian profile; "same" mode ensures
    # it is the same length as the original signal for easy indexing
    corr_signal = np.correlate(signal, temp_y, mode="same")
    return corr_signal
    

def interp_mmw(files, nelements=10000, window_function=None, cutoff=None):
    """
        Function that will process archival millimeter-wave data
        by providing a list of filepaths.
        
        The function will parse the data out of each file, and
        produce a plot that is stitched together.
        
        The first step is to read in all the data, and append
        the parsed data into full_freq and full_int, corresponding
        to the experimental frequency and intensities. This is
        used to then determine the limits to the x axis for plotting.
        
    """
    full_freq = list()
    full_int = list()
    
    print("Parsing files.")
    for index, file in enumerate(files):
        frequency, intensity, settings = parse_data(file)
        intensity = fft_routines.fft_filter(intensity, window_function, cutoff)
        if index == 0:
            minx = np.min(frequency)
            maxx = np.max(frequency)
        else:
            curr_minx = np.min(frequency)
            curr_maxx = np.max(frequency)
            if curr_minx < minx:
                minx = curr_minx
            if curr_maxx > maxx:
                maxx = curr_maxx
        full_freq.append(frequency)
        full_int.append(intensity)
    """
        Stitching the spectrum together using a weighted
        Shepard interpolation.
    """
    print("Lowest frequency: " + str(minx))
    print("Highest frequency: " + str(maxx))
    print("Performing Shepard interpolation")
    frequency = np.linspace(minx, maxx, nelements)
    interp_y = np.zeros(nelements)
    
    # Loop over each frequency bin
    for index, interp_freq in enumerate(frequency):
        if (index / len(frequency)) > 0.5:
            print("50% done.")
        # Calculate the Shepard interpolation at the given frequency point
        interp_y[index] = interpolation.eval_shep_interp(full_freq, full_int, interp_freq, p=16.)
    
    df = pd.DataFrame(data=list(zip(frequency, interp_y)), columns=["Frequency", "Intensity"])
    return df


def batch_interpolate(files, datapoints=50000, zeeman_on=None, window_function=None, pass_filter=None):
    """
        Function for batch processing multiple scan files, and concetanates them
        together to generate a single, interactive spectrum.
        
        Takes input argument files as a list of filepaths corresponding to scans
        that are (generally) without Zeeman. The optional argument, zeeman_on,
        is a list of filepaths corresponding to scans with Zeeman on.
        
        All of the scans will be processed with the same window functions and
        cut offs, so be sure to optimize them before batch processing.
        
        Returns a dataframe with up to three columns: Zeeman off/on, and off - on.
    """
    spectrum_df = interp_mmw(files, datapoints, window_function, pass_filter)
    # If Zeeman on spectra are provided, we process those too, and perform subtraction
    if zeeman_on is not None:
        zeeman_on_df = interp_mmw(zeeman_on, datapoints, window_function, pass_filter)
        spectrum_df.columns = ["Frequency", "Zeeman OFF"]
        spectrum_df["Zeeman ON"] = zeeman_on_df["Intensity"].values()
        spectrum_df["OFF - ON"] = spectrum_df["Zeeman OFF"] - spectrum_df["Zeeman ON"]
    
    fig = plot_spectrum(spectrum_df)
    return fig, df


def batch_shepard(files, stepsize=0.1, xrange=10., p=4., threshold=0.1, npoints=15, window_function=None, pass_filter=None):
    """
        Routine for performing mass interpolation using Shepard's interpolation method
        on mmw data. Input is a list of files, and parameters used to interpolate and
        to filter the data.
        
        Stepsize is given in units of MHz.
    """
    print("Loading data...")
    datalist = [
        open_mmw(file, window_function, pass_filter) for file in files
    ]
    print("Load complete.")
    min_freq = 1e9
    max_freq = 0.
    print("Finding frequency range")
    for data in datalist:
        current_min = data["Frequency"].min()
        current_max = data["Frequency"].max()
        if current_min < min_freq:
            min_freq = current_min
        if current_max > max_freq:
            max_freq = current_max
    print("Max freq: " + str(max_freq))
    print("Min freq: " + str(min_freq))
    # Initialize the new frequency range in one MHz steps
    xnew = np.arange(min_freq, max_freq, stepsize)
    new_df = pd.DataFrame(data=xnew, columns=["Frequency"])
    print("Performing interpolation")
    for column in ["Zeeman OFF", "Zeeman ON"]:
        print(column)
        new_df[column] = interpolation.calc_shep_interp(
            datalist,
            xnew,
            column,
            p,
            threshold,
            npoints
        )
    print("Interpolation done")
    new_df["OFF - ON"] = new_df["Zeeman OFF"] - new_df["Zeeman ON"]
    return new_df  


def batch_plot(files, window_function=None, pass_filter=None):
    for index, file in enumerate(files):
        if index == 0:
            full_df = open_mmw(file, window_function, pass_filter)
        else:
            full_df = pd.concat([
                full_df,
                open_mmw(file, window_function, pass_filter)
            ])
    full_df = full_df.sort_values(["Frequency"], ascending=True)
    fig = plot_spectrum(full_df)
    
    return full_df, fig


def fit_spectrum_chunk(dataframe: pd.DataFrame, freq_range: List[float], frequencies: np.ndarray, **kwargs):
    """
    Automates the fitting of multiple frequencies to a single spectrum.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pandas DataFrame containing the millimeter-wave spectrum.
    freq_range : 2-tuple
        Iterable with two floats, corresponding to the limits of the spectrum
        (in frequency) we want to consider for the fit.
    frequencies : np.ndarray
        NumPy 1D array containing center frequencies to fit.

    Returns
    -------
    [type]
        [description]
    """
    freq_range = sorted(freq_range)
    sliced_df = dataframe.loc[dataframe["Frequency"].between(*freq_range)]
    sliced_df["XCorrelation"] = cross_correlate_lorentzian(
        sliced_df["Field OFF"], **kwargs
        )
    param_dict = dict()
    # Build fitting objective function from sum of second derivative
    # profiles
    for index, frequency in enumerate(frequencies):
        curr_model = lmfit.models.Model(
            sec_deriv_lorentzian, prefix=f"Line{index}_"
        )
        curr_params = curr_model.make_params()
        curr_params[f"Line{index}_x0"].set(value=frequency - 0.3, min=frequency - 15., max=frequency + 15.)
        curr_params[f"Line{index}_I"].set(value=-1., min=-20., max=-0.05)
        curr_params[f"Line{index}_gamma"].set(value=0.25, min=0.1, max=0.4)
        if index == 0:
            model = curr_model
            parameters = curr_params
        else:
            model += curr_model
            parameters += curr_params
            
    # Fit the cumulative spectrum 
    fit_results = model.fit(
        sliced_df["XCorrelation"],
        parameters,
        x=sliced_df["Frequency"]
    )
    sliced_df["Fit"] = fit_results.eval()

    # print("Results of the fit:")
    # for param, covar in zip(fit_results.best_values, np.sqrt(np.diag(fit_results.covar))):
    #     print(param + ": " + str(fit_results.best_values[param]) + "  +/-  " + str(covar))
    
    return fit_results, sliced_df


def fit_lines(dataframe, frequencies, name, window_length=10., ycol="Field off", off_dataframe=None):
    """ Wrapper for the fit_chunk function, for when you want to
        try and fit many lines in a catalog.
        
        The input arguments are the survey dataframe, a list of
        center frequencies you want to try and fit, the name of
        the species. Optional argument is the name of column you
        want to use for the intensities.
        
        The function will iterate through each center frequency
        and try to fit it.
    """
    foldername = Path(f"../data/clean/mmw_fits/{name}")
    try:
        os.mkdir(foldername)
    except FileExistsError:
        pass
    
    successes = list()
    fit_freq = list()
    amplitudes = list()
    uncertainties = list()
    
    # Loop over each catalog frequency
    for index, frequency in enumerate(frequencies):
        
        # Slice out a 20 MHz window - this should be sufficiently
        # large to capture uncertainty in lines
        min_freq = frequency - window_length
        max_freq = frequency + window_length
        # Attempt to fit the window
        fit, spec_df = fit_spectrum_chunk(
            dataframe,
            min_freq,
            max_freq,
            [frequency],
        )
        fit_freq.append(
            np.round(fit.best_values["center0"], decimals=4)
        )
        amplitudes.append(
            np.round(fit.best_values["A0"], decimals=4)
        )
        uncertainties.append(
            np.round(np.sqrt(np.diag(fit.covar))[-1], decimals=4)
        )
        successes.append(True)
    package = list(zip(frequencies, successes, fit_freq, uncertainties, amplitudes))
    fit_df = pd.DataFrame(
        data=package, 
        columns=["Catalog frequency", "Success", "Fitted frequency", "Uncertainty", "Amplitude"]
    )
    
    fit_df["Calc. vs Obs."] = fit_df["Catalog frequency"] - fit_df["Fitted frequency"]
    
    fit_df.to_csv(foldername + "/fit_results.csv")
    
    # Make the static comparison
    if off_dataframe:
        fig, axarray = static_comparison(
            fit_df["Fitted frequency"].values,
            dataframe,
            off_dataframe,
        )
        fig.savefig(foldername + "/on-off-comparison.pdf", dpi=300, format="pdf")
    
    return fit_df
            
        

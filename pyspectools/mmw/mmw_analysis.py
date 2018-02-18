
import numpy as np
import lmfit
import pandas as pd
import os

from .fft_routines import *
from .interpolation import *
from .plotting_func import *

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
                settings["Date"] = str(scan_details[4])
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
    start_freq = settings["Frequency"] - (settings["Frequency step"] * settings["Points"] / 2)
    end_freq = settings["Frequency"] + (settings["Frequency step"] * settings["Points"] / 2)
    frequency = np.arange(start_freq, end_freq, settings["Frequency step"])

    return frequency, fieldoff_intensities, fieldon_intensities, settings


def open_mmw(filepath, window_function=None, pass_filter=None):
    """
        Function for opening and processing a single millimeter
        wave data file.
        
        Returns a pandas dataframe containing Frequency and
        Intensity information.
    """
    frequency, fieldoff_intensities, fieldon_intensities, settings = parse_data(filepath)
    sample_rate = 1. / settings["Frequency step"] * 1e6
    
    
    fieldoff_intensities = fft_filter(fieldoff_intensities, window_function, pass_filter, sample_rate=sample_rate)
    fieldon_intensities = fft_filter(fieldon_intensities, window_function, pass_filter, sample_rate=sample_rate)
    
    try:
        intensity = fieldoff_intensities - fieldon_intensities
    except ValueError:
        print(filepath + " had error loading.")
    mmw_df = pd.DataFrame(
        data=list(zip(frequency, fieldoff_intensities, fieldon_intensities, intensity)),
        columns=["Frequency", "Zeeman OFF", "Zeeman ON", "OFF - ON"]
    )
    return mmw_df


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
        intensity = fft_filter(intensity, window_function, cutoff)
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
        interp_y[index] = eval_shep_interp(full_freq, full_int, interp_freq, p=16.)
    
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
        new_df[column] = calc_shep_interp(
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


def fit_chunk(dataframe, min_freq, max_freq, frequencies, ycol="Field off"):
    """
        Function that will fit second-derivative lineprofiles to chunks
        of a spectrum.
        
        The full spectrum is provided as a dataframe, and uses pandas
        to slice the minimum and maximum frequencies. The frequencies
        are provided as a list of center frequencies to fit.
        
        An optional argument, ycol, is used to specify which dataframe
        column to use.
    """
    sliced_df = dataframe.loc[(dataframe["Frequency"] > min_freq) & (dataframe["Frequency"] < max_freq)]
    exp_string = "(-2. * A{index} * width{index}**3.) / (pi * (width{index}**2. + (x - center{index})**2.)**2.) \
             + (8. * A{index} * width{index}**3. * (x - center{index})**2.) / (pi * (width{index}**2. + (x - center{index})**2.)**3.)"
    
    # Build fitting objective function from sum of second derivative
    # profiles
    for index, frequency in enumerate(frequencies):
        curr_model = lmfit.models.ExpressionModel(
            exp_string.format_map({"index": str(index)})
        )
        # Set up a dictionary with parameters
        string_params = {
            "width" + str(index): 1.,
            "A" + str(index): 1.,
            "center" + str(index): frequency
        }
        # Build parameters
        curr_parameters = curr_model.make_params(**string_params)
        # Constrain amplitude to be positive - no absorption!
        curr_parameters["A" + str(index)].set(1., min=0.)
        if index == 0:
            model = curr_model
            parameters = curr_parameters
        else:
            model+=curr_model
            parameters+=curr_parameters
    # Fit the peaks
    fit_results = model.fit(
        sliced_df[ycol],
        parameters,
        x=sliced_df["Frequency"]
    )
    sliced_df["Fit"] = fit_results.eval()
    
    fig = plot_spectrum(sliced_df)
    
    print("Results of the fit:")
    for param, covar in zip(fit_results.best_values, np.sqrt(np.diag(fit_results.covar))):
        print(param + ": " + str(fit_results.best_values[param]) + "  +/-  " + str(covar))
    
    return fit_results, fig


def fit_lines(dataframe, frequencies, name, window_length=10., ycol="Field off", off_dataframe=None):
    """
        Wrapper for the fit_chunk function, for when you want to
        try and fit many lines in a catalog.
        
        The input arguments are the survey dataframe, a list of
        center frequencies you want to try and fit, the name of
        the species. Optional argument is the name of column you
        want to use for the intensities.
        
        The function will iterate through each center frequency
        and try to fit it.
    """ 
    foldername = "../data/mmw_fits/" + name + "_" + ycol.replace(" ", "")
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
        try:
            fit, fig = fit_chunk(
                dataframe,
                min_freq,
                max_freq,
                [frequency],
                ycol
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
            save_plot(fig, foldername + "/peak" + str(index) + ".html")
        except:
            successes.append(False)
            fit_freq.append(None)
            amplitudes.append(0.)
            uncertainties.append(None)
    package = list(zip(frequencies, successes, fit_freq, uncertainties, amplitudes))
    fit_df = pd.DataFrame(
        data=package, 
        columns=["Catalog frequency", "Success", "Fitted frequency", "Uncertainty", "Amplitude"]
    )
    
    fit_df["Calc. vs Obs."] = fit_df["Catalog frequency"] - fit_df["Fitted frequency"]
    
    fit_df.to_csv(foldername + "/fit_results.csv")
    
    # Make the static comparison
    if off_dataframe is not None:
        fig, axarray = static_comparison(
            fit_df["Fitted frequency"].values,
            dataframe,
            off_dataframe,
        )
        fig.savefig(foldername + "/on-off-comparison.pdf", dpi=300, format="pdf")
    
    return fit_df
            
        

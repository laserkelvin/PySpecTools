from pyspectools import parsecat as pc
import pandas as pd
import numpy as np
import peakutils
from matplotlib import pyplot as plt
from matplotlib import colors
from plotly.offline import plot, init_notebook_mode, iplot
from plotly import tools
import plotly.graph_objs as go
from scipy import signal as spsig
from itertools import combinations


def parse_specdata(filename):
    # For reading the output of a SPECData analysis
    return pd.read_csv(filename, skiprows=4)


def parse_spectrum(filename, threshold=20.):
    """ Function to read in a blackchirp or QtFTM spectrum from file """
    dataframe = pd.read_csv(
        filename, delimiter="\t", names=["Frequency", "Intensity"], skiprows=1
    )
    return dataframe[dataframe["Intensity"] <= threshold]


def center_cavity(dataframe, thres=0.3, verbose=True):
    """ Finds the center frequency of a Doppler pair in cavity FTM measurements
        and provides a column of offset frequencies.

        Sometimes the peak finding threshold has to be tweaked to get the center
        frequency correctly.
    """
    # Find the peak intensities
    center_indexes = peakutils.indexes(dataframe["Intensity"], thres=thres)
    peak_frequencies = dataframe.iloc[center_indexes]["Frequency"]
    # Calculate the center frequency as the average
    center = np.average(peak_frequencies)
    if verbose is True:
        print("Center frequency at " + str(center))
    dataframe["Offset Frequency"] = dataframe["Frequency"] - center


def configure_colors(dataframe):
    """ Generates color palettes for plotting arbitrary number of SPECData
        assignments.
    """
    num_unique = len(dataframe["Assignment"].unique())
    return plt.cm.spectral(np.linspace(0., 1., num_unique))


def plot_specdata_mpl(dataframe):
    """ Function to display SPECData output using matplotlib.
        The caveat here is that the plot is not interactive, although it does
        provide an overview of what the assignments are. This is probably the
        preferred way if preparing a plot for a paper.
    """
    colors = configure_colors(dataframe)
    fig, exp_ax = plt.subplots(figsize=(10,6))

    exp_ax.vlines(dataframe["Exp. Frequency"], ymin=0., ymax=dataframe["Exp. Intensity"], label="Observations")

    exp_ax.set_yticks([])
    exp_ax.set_xlabel("Frequency (MHz)")

    assign_ax = exp_ax.twinx()
    current_limits = assign_ax.get_xlim()
    for color, assignment in zip(colors, dataframe["Assignment"].unique()):
        trunc_dataframe = dataframe[dataframe["Assignment"] == assignment]
        assign_ax.vlines(
            trunc_dataframe["Frequency"],
            ymin=np.negative((trunc_dataframe["Intensity"] / trunc_dataframe["Intensity"].max())),
            ymax=0.,
            alpha=0.5,
            label=assignment,
            color=color
        )
    exp_ax.hlines(0., 0., 30000.,)
    assign_ax.set_ylim([1., -1.])
    exp_ax.set_ylim([1., -1.])
    assign_ax.set_xlim(current_limits)

    assign_ax.set_yticks([])
    assign_ax.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=True)


def plot_specdata_plotly(dataframe, output="specdata_interactive.html"):
    """ Interactive SPECData result plotting using plotly.
        The function will automatically swap between spectra and peaks by
        inspecting the number of data points we have.
    """
    init_notebook_mode(connected=False)
    if len(dataframe) >= 10000:
        exp_plot_function = go.Scatter
    else:
        exp_plot_function = go.Bar
    plots = list()
    plots.append(
        # Plot the experimental data
        exp_plot_function(
            x = dataframe["Exp. Frequency"],
            y = dataframe["Exp. Intensity"],
            name = "Experiment",
            width = 1.,
            opacity = 0.6
        )
    )
    # Use Matplotlib function to generate a colourmap
    color_palette = configure_colors(dataframe)
    # Loop over the colours and assignments
    for color, assignment in zip(color_palette, dataframe["Assignment"].unique()):
        # Truncate the dataframe to only hold the assignment
        trunc_dataframe = dataframe[dataframe["Assignment"] == assignment]
        plots.append(
            go.Bar(
                x=trunc_dataframe["Frequency"],
                y=np.negative((trunc_dataframe["Intensity"] / trunc_dataframe["Intensity"].max())),
                name=assignment,
                width=1.,
                marker={
                    # Convert the matplotlib color array to hex code
                    "color": colors.rgb2hex(color[:-1])
                }
            )
        )
    layout = go.Layout(
        yaxis={"title": "Intensity"},
        xaxis={"title": "Frequency (MHz)"}
    )
    fig = go.Figure(data=plots, layout=layout)
    plot(fig, filename=output)


class Scan:
    """ Object for analyzing raw FIDs from QtFTM.
        The goal is to be able to retrieve a scan file from QtFTM, perform the
        FFT analysis and everything externally, without the need to open QtFTM.

        Some of the methods I intend to implement are:

        1. Doppler fitting
        2. Export the resulting FFT for external plotting
        3. Stock plotting
    """
    def __init__(self, scan_id, window_function=None):
        self.settings = dict()
        self.fid = list()
        with open(str(scan_id) + ".txt") as fid_file:
            self.parsed_data = fid_file.readlines()
        self.read_fid_settings()
        self.fid = np.array(self.fid)

    def read_fid_settings(self):
        """ Function to read in the settings in an FID from QtFTM """
        read_fid = False
        for line in self.parsed_data:
            split_line = line.split()
            if len(split_line) != 0:
                if "#" in split_line[0]:
                    # Settings lines are commented with "#"
                    if "Scan" in line:
                        # Read in the scan ID
                        self.settings["ID"] = "-".join(split_line[1:])
                    elif "Shots" in line:
                        # Read in the number of Shots
                        self.settings["Shots"] = int(split_line[1])
                    elif "Cavity freq" in line:
                        # Read in the cavity frequency; units of MHz
                        self.settings["Frequency"] = float(split_line[2])
                    elif "Tuning Voltage" in line:
                        # Read in tuning voltage; units of mV
                        self.settings["Tuning voltage"] = int(split_line[2])
                    elif "Attenunation" in line:
                        # Read in the Attenuation; units of dB
                        self.settings["Attenuation"] = int(split_line[1])
                    elif "Cavity Voltage" in line:
                        # Read in cavity voltage; units of mV
                        self.settings["Cavity voltage"] = int(split_line[2])
                    elif "FID spacing" in line:
                        # Read in the FID spacing, units of seconds
                        self.settings["FID spacing"] = float(split_line[2])
                    elif "FID points" in line:
                        # Read in the number of points we expect for the FID
                        self.settings["FID points"] = int(split_line[2])
                    elif "Probe" in line:
                        # The probe frequency, in MHz
                        self.settings["Probe frequency"] = float(split_line[2])
                if read_fid is True:
                    self.fid.append(float(line))
                if "fid" in line:
                    # Start reading the FID in after this line is found
                    read_fid = True
            else:
                pass

    def fid2fft(self, window_function=None, dc_offset=True, exp_filter=None):
        """ Perform the DFT of an FID using NumPy's FFT package. """
        available_windows = [
            "blackmanharris",
            "blackman",
            "boxcar",
            "gaussian",
            "hanning",
            "bartlett"
        ]
        if window_function is not None:
        # Use a scipy.signal window function to process the FID signal
            if window_function not in available_windows:
                print("Incorrect choice for window function.")
                print("Available:")
                print(available_windows)
            else:
                self.fid *= spsig.get_window(window_function, self.fid.size)
        # Perform the FFT
        if exp_filter is not None:
            self.fid *= spsig.exponential(self.fid.size, exp_filter)
        amplitude = np.fft.fft(self.fid)
        # Calclate the frequency window
        frequency = np.linspace(
            self.settings["Probe frequency"],
            self.settings["Probe frequency"] + 1.,
            amplitude.size
        )
        self.fft = pd.DataFrame(
            data=list(zip([frequency, amplitude])),
            columns=["Frequency (MHz)", "Intensity (V)"]
        )


def generate_ftb_line(frequency, shots, **kwargs):
    """ Function that generates an FTB file for a list of
        frequencies, plus categorization tests.
    """
    line = "ftm:{:.4f} shots:{}".format(frequency, shots)
    for key, value in kwargs.items():
        line+=" {}:{}".format(key, value)
    line+="\n"
    return line


def categorize_frequencies(frequencies, nshots=50, intensities=None, power=None, attn_list=None,
        dipole=None, attn=None, magnet=False, dr=False, discharge=False):
    """
        Function that will format an FT batch file to perform categorization
        tests, with some flexibility on how certain tests are performed.
    """
    ftb_str = ""
    if intensities is None:
        shots = np.full(len(frequencies), nshots, dtype=int)
    else:
        shots = np.sqrt(nshots / intensities).astype(int)

    if dipole:
        if attn is None:
            # If dipole test requested, but no attenuation
            # supplied do the default sweep
            dipole_test = [0.01, 0.1, 1.0, 3.0, 5.0]
            dipole_flag = "dipole"
        else:
            # Otherwise run specific attenuations
            dipole_test = attn_list
            dipole_flag = "atten"

    if dr is True:
        freq_list = combinations(frequencies, 2)
        print(list(freq_list))
    else:
        freq_list = frequencies

    # loop over each frequency and number of shots
    for value, shotcount in zip(freq_list, shots):
        if dr is True:
            freq, dr_freq = value
        else:
            freq = value
            # Generate normal observation
        try:
            freq = float(freq)
            shotcount = int(shotcount)
            if dr is True:
                dr_freq = float(dr_freq)

            ftb_str+=generate_ftb_line(
                freq, 
                shotcount, 
                **{"skiptune": "false"})

            if dr is True:
                ftb_str+=generate_ftb_line(freq, 
                    shotcount, 
                    **{
                        "skiptune": "true",
                        "drfreq": dr_freq,
                        "drpower": "10"
                    }
                    )

            if dipole is True:
                for dipole_value in dipole_test:
                    ftb_str+=generate_ftb_line(freq, shotcount, **{dipole_flag: dipole_value})

            if magnet is True:
                ftb_str+=generate_ftb_line(freq, shotcount, **{"magnet": "true"})

            if discharge is True:
                # Toggle the discharge stack on and off
                ftb_str+=generate_ftb_line(freq, shotcount, **{"pulse,1,enabled": "false"})
                ftb_str+=generate_ftb_line(freq, shotcount, **{"pulse,1,enabled": "true"})
        except ValueError:
            print("Error with " + str(value))

    return ftb_str

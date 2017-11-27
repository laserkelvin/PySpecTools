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
from scipy import constants

init_notebook_mode(connected=False)


def dop2freq(velocity, frequency):
    # Frequency given in MHz, Doppler_shift given in km/s
    # Returns the expected Doppler shift in frequency (MHz)
    return ((velocity * 1000. * frequency) / constants.c)

def freq2vel(frequency, offset):
    # Takes the expected Doppler contribution to frequency and the rest
    # frequency, and returns the Doppler shift in km/s
    return ((constants.c * offset) / frequency) / 1000.


def parse_specdata(filename):
    # For reading the output of a SPECData analysis
    return pd.read_csv(filename, skiprows=4)


def parse_spectrum(filename, threshold=20.):
    """ Function to read in a blackchirp or QtFTM spectrum from file """
    dataframe =  pd.read_csv(
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


def plot_chirp(chirpdf, catfiles=None):
    """ Function to perform interactive analysis with a chirp spectrum, as well
        as any reference .cat files you may want to provide.
        This is not designed to replace SPECData analysis, but simply to perform
        some interactive viewing of the data.

        The argument `catfiles` is supplied as a dictionary; where the keys are
        the names of the species, and the items are the paths to the .cat files
    """

    # Generate the experimental plot first
    plots = list()
    exp_trace = go.Scatter(
        x = chirpdf["Frequency"],
        y = chirpdf["Intensity"],
        name = "Experiment"
    )

    plots.append(exp_trace)
    if catfiles is not None:
        # Generate the color palette, and remove the alpha value from RGBA
        color_palette = plt.cm.spectral(np.linspace(0., 1., len(catfiles)))[:,:-1]
        # Loop over each of the cat files
        for color, species in zip(color_palette, catfiles):
            species_df = pc.pick_pickett(catfiles[species])
            plots.append(
                go.Bar(
                    x = species_df["Frequency"],
                    y = species_df["Intensity"] / species_df["Intensity"].min(),
                    name = species,
                    marker = {
                        # Convert the matplotlib rgb color to hex code
                        "color": colors.rgb2hex(color)
                    },
                    width = 1.,
                    opacity = 0.6,
                    yaxis = "y2"
                )
            )
    layout = go.Layout(
        autosize=False,
        height=600,
        width=900,
        xaxis={"title": "Frequency (MHz)"},
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
        yaxis={"title": ""},
        yaxis2={"title": "", "side": "right", "overlaying": "y", "range": [0., 1.]}
    )
    fig = go.Figure(data=plots, layout=layout)
    iplot(fig)

    return fig


def stacked_plot(dataframe, frequencies, freq_range=0.01):
    # Function for generating an interactive stacked plot.
    # This form of plotting helps with searching for vibrational satellites.
    # Input is the full dataframe containing the frequency/intensity data,
    # and frequencies is a list containing the centering frequencies.
    # The frequency range is specified as a percentage of frequency, so it
    # will change the range depending on what the centered frequency is.
    nplots = len(frequencies)

    plot_func = go.Scatter

    fig = tools.make_subplots(
        rows=nplots,
        cols=1,
        specs=[[{}] for plot in range(nplots)],
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=tuple("{:.2f} MHz".format(frequency) for frequency in frequencies),
    )

    frequencies = np.sort(frequencies)[::-1]
    print(frequencies)

    for index, frequency in enumerate(frequencies):
        dataframe["Offset " + str(index)] = dataframe["Frequency"] - frequency
        freq_cutoff = freq_range * frequency
        sliced_df = dataframe.loc[
            (dataframe["Offset " + str(index)] > -freq_cutoff) & (dataframe["Offset " + str(index)] < freq_cutoff)
        ]
        trace = plot_func(
            x=sliced_df["Offset " + str(index)],
            y=sliced_df["Intensity"],
            text=sliced_df["Frequency"],
            mode="lines"
        )
        # Plotly indexes from one because they're stupid
        fig.append_trace(trace, index + 1, 1)
        fig["layout"]["xaxis1"].update(range=[-freq_cutoff,freq_cutoff], title="Offset frequency (MHz)", showgrid=True)
        fig["layout"]["yaxis" + str(index + 1)].update(showgrid=False)
    fig["layout"].update(
        autosize=False,
        height=800,
        width=1000,
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
        showlegend=False
    )

    iplot(fig)

    return fig


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

class scan:
    """ Object for analyzing raw FIDs from QtFTM.
        The goal is to be able to retrieve a scan file from QtFTM, perform the
        FFT analysis and everything externally, without the need to open QtFTM.

        Some of the methods I intend to implement are:

        1. Doppler fitting
        2. Export the resulting FFT for external plotting
        3. Stock plotting
    """
    def __init___(self, id, window_function=None):
        self.settings = dict()
        self.fid = list()
        with open(str(id) + ".txt") as fid_file:
            self.parsed_data = fid_file.readlines()
        self.read_fid_settings()
        self.fid = np.array(self.fid)

    def read_fid_settings(self):
        """ Function to read in the settings in an FID from QtFTM """
        for line in self.parsed_data:
            split_line = line.split()
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
                elif "Tuning" in line:
                    # Read in tuning voltage; units of mV
                    self.settings["Tuning voltage"] = int(split_line[2])
                elif "Attenunation" in line:
                    # Read in the Attenuation; units of dB
                    self.settings["Attenuation"] = int(split_line[1])
                elif "Cavity voltage" in line:
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
            data=[frequency, amplitude],
            columns=["Frequency (MHz)", "Intensity (V)"]
        )

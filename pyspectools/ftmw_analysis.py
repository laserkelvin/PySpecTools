
from itertools import combinations, product
import os

import pandas as pd
import numpy as np
import peakutils
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import signal as spsig
from plotly.offline import plot, init_notebook_mode, iplot
from plotly import tools
import plotly.graph_objs as go

from pyspectools import parsecat as pc


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

        kwargs are passed as additional options for the ftb
        batch. Keywords are:

        magnet: bool
        dipole: float
        atten: int
        skiptune: bool
        drfreq: float
        drpower: int

        parameters:
        ---------------
        frequency - float for frequency in MHz
        shots - int number of shots to integrate for

        returns:
        ---------------
        ftbline - str
    """
    line = "ftm:{:.4f} shots:{}".format(frequency, shots)
    for key, value in kwargs.items():
        line+=" {}:{}".format(key, value)
    line+="\n"
    return line


def neu_categorize_frequencies(frequencies, intensities=None, nshots=50, **kwargs):
    """
        Routine to generate an FTB batch file for performing a series of tests
        on frequencies.
    """
    ftb_string = ""
    if intensities:
        norm_int = intensities / np.max(intensities)
        shotcounts = np.round(nshots / norm_int).astype(int)
    else:
        shotcounts = np.full(len(frequencies), nshots, dtype=int)

    # default settings for all stuff
    param_dict = {
        "dipole": 1.,
        "magnet": "false",
        "drpower": "10",
        "skiptune": "false"
        }

    param_dict.update(kwargs)
    for freq, shot in zip(frequencies, shotcounts):
        ftb_string+=generate_ftb_str(freq, shot, **param_dict)
        if "magnet" in kwargs:
            param_dict["magnet"] = "true"
            ftb_string+=generate_ftb_str(freq, shot, **param_dict)


def categorize_frequencies(frequencies, nshots=50, intensities=None, 
        power=None, attn_list=None, dipole=None, attn=None, 
        magnet=False, dr=False, discharge=False):
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

def calculate_integration_times(intensity, nshots=50):
    """
        Method for calculating the expected integration time
        in shot counts based on the intensity; either theoretical
        line strengths or SNR.

        parameters:
        ---------------
        intensity - array of intensity metric; e.g. SNR
        nshots - optional int number of shots used for the strongest line

        returns:
        ---------------
        shot_counts - array of shot counts for each frequency
    """
    norm_int = intensity / np.max(intensity)
    shot_counts = np.round(nshots / norm_int).astype(int)
    return shot_counts

class AssayBatch:
    @classmethod
    def from_csv(cls, filepath, exp_id):
        df = pd.read_csv(filepath)
        return cls(df, exp_id)

    def __init__(self, uline_dataframe, exp_id, 
            freq_col="Frequency", int_col="Intensity"):
        folders = ["assays", "ftbfiles"]
        for folder in folders:
            if os.path.isdir(folder) is False:
                os.mkdir(folder)

        if os.path.isdir("assays/plots") is False:
            os.mkdir("./assays/plots")

        self.data = uline_dataframe
        self.exp_id = exp_id

        if freq_col not in self.data.columns:
            self.freq_col = self.data.columns[0]
        else:
            self.freq_col = freq_col
        if int_col not in self.data.columns:
            self.int_col = self.data.columns[0]
        else:
            self.int_col = int_col

    def calc_scan_SNR(self, peak_int, noise_array):
        """
            Calculate the signal-to-noise ratio of a peak
            for a given reference noise intensity array
            and peak intensity.
        """
        noise = np.mean(noise_array)
        return peak_int / noise

    def dipole_analysis(self, batch_path, thres=0.5,
            dipoles=[1., 0.01, 0.1, 1.0, 3.0, 5.], snr_thres=5.):
        """
            Method for determining the optimal dipole moment to use 
            for each frequency in a given exported batch of dipole tests.

            In addition to returning a pandas dataframe, it is also
            stored as the object attribute dipole_df.

            parameters:
            ----------------
            batch_path - filepath to the exported XY file from a QtFTM batch
            thres - threshold in absolute intensity for peak detection (in Volts)
            dipoles - list of dipole moments used in the screening
            snr_thres - threshold in signal-to-noise for line detection

            returns:
            ----------------
            optimal_df - pandas dataframe containing the optimal dipole moments
        """
        batch_df = pd.read_csv(batch_path, sep="\t")
        batch_df.columns = ["Scan", "Intensity"]
        
        # Reference scan numbers from the batch
        scan_numbers = np.unique(np.around(batch_df["Scan"]).astype(int))
        # Make a dataframe for what we expect everything should be
        full_df = pd.DataFrame(
            data=list(product(self.data["Frequency"], dipoles)),
            columns=["Frequency", "Dipole"]
            )
        full_df["Scan"] = scan_numbers
        
        # Loop over each scan, and determine whether or not there is a peak
        # If there is sufficiently strong feature, add it to the list with the
        # scan number
        detected_scans = list()
        for index, scan in enumerate(scan_numbers):
            scan_slice = batch_df.loc[
                (batch_df["Scan"] >= scan - 0.5) & (batch_df["Scan"] <= scan +0.5)
                ]
            # Find the peaks based on absolute signal intensity
            # Assumption is that everything is integrated for the same period of time
            peaks = scan_slice.iloc[
                peakutils.indexes(scan_slice["Intensity"], thres=thres, thres_abs=True)
                ].sort_values(["Intensity"], ascending=False)

            peak_int = np.average(peaks["Intensity"][:2])
            snr = self.calc_scan_SNR(peak_int, scan_slice["Intensity"][-10:])
            if snr >= snr_thres:
                detected_scans.append([scan, snr])
        snr_df = pd.DataFrame(data=detected_scans, columns=["Scan", "SNR"])
        # Merge the dataframes based on scan number. This will identify
        # scans where we observe a line
        obs_df = full_df.merge(snr_df, on=["Scan"], copy=False)

        optimal_data = list()
        # Loop over each frequency in order to determine the optimal
        # dipole moment to use
        for frequency in np.unique(obs_df["Frequency"]):
            slice_df = obs_df.loc[
                obs_df["Frequency"] == frequency
                ]
            # Sort the best response dipole moment at the top
            slice_df.sort_values(["SNR"], inplace=True, ascending=False)
            slice_df.index = np.arange(len(slice_df))
            optimal_data.append(slice_df.iloc[0].values)
        optimal_df = pd.DataFrame(
            optimal_data,
            columns=["Frequency", "Dipole", "Scan", "SNR"]
            )
        optimal_df.sort_values(["SNR"], ascending=False, inplace=True)

        self.dipole_df = optimal_df

        # Generate histogram of dipole moments
        with plt.style.context("publication"):
            fig, ax = plt.subplots()

            self.dipole_df["Dipole"].hist(ax=ax)
            ax.set_xlabel("Dipole (D)")
            ax.set_ylabel("Counts")

            fig.savefig(
                "./assays/plots/{}-dipole.pdf".format(self.exp_id),
                format="pdf",
                transparent=True
                )
        return optimal_df

    def generate_magnet_test(self, dataframe=None, cal=False, nshots=50, **kwargs):
        """
            Generate an FT batch file for performing magnet tests.
        """
        if dataframe is None:
            dataframe = self.dipole_df
            dataframe["Shots"] = calculate_integration_times(
                dataframe["SNR"].values,
                nshots
                )
        ftb_str = ""
        
        # If there are no shots determined, 
        if "Shots" not in dataframe:
            dataframe["Shots"] = nshots

        if cal_freq is True:
            cal_settings = {
                "cal_freq": dataframe["Frequency"][0],
                "cal_rate": 100,
                "cal_shots": dataframe["Shots"][0]
                }
            if "Dipole" in dataframe:
                cal_settings["cal_dipole"] = dataframe["Dipole"][0]
            cal_settings.update(kwargs)
            # Generate the calibration string
            cal_str = generate_ftb_line(
                cal_settings["cal_freq"],
                cal_settings["cal_shots"],
                **{
                    "dipole": cal_settings["cal_dipole"],
                    "skiptune": "false"
                }
                )
            cal_str = cal_str.replace("\n", " cal\n")

        # generate the batch file
        for index, row in dataframe.iterrows():
            # If we want to add a calibration line
            if index % cal_settings["cal_rate"] == 0:
                ftb_str+=cal_str
            param_dict = {}
            # Make sure the magnet is off
            if "Dipole" in row:
                param_df["dipole"] = row["Dipole"]
            param_df["magnet"] = "false"
            param_df["skiptune"] = "false"
            ftb_str+=generate_ftb_line(
                row["Frequency"],
                row["Shots"],
                **param_df
                )
            # Turn on the magnet
            param_df["skiptune"] = "true"
            param_df["magnet"] = "true"
            ftb_str+=generate_ftb_line(
                row["Frequency"],
                row["Shots"],
                **param_df
                )

        with open("./ftbfiles/{}.ftb".format(self.exp_id), "w+") as write_file:
            write_file.write(ftb_str)

#    def generate_dr_test(self, cluster_dict=None, nshots=50):
#        """
#            Take the dipole moment dataframe and generate a DR batch.
#            If possible, we will instead use the progressions predicted
#            by the cluster model.
#        """
#

    def plot_scan(self, scan_number):
        """
            Quick method for plotting up a strip to highlight a particular
            scan in a batch.

            parameters:
            ---------------
            scan_number - float corresponding to the scan 
        """
        fig = go.FigureWidget()

        fig.add_scatter(
            x=self.data["Scan"],
            y=self.data["Intensity"]
            )

        fig.add_bar(
            x=[scan_number],
            y=[np.max(self.data["Intensity"])]
            )

        return fig

    def static_plot(self, scan_number, dataframe=None):
        """
            Produce a static plot with matplotlib of a particular
            scan from a batch.

            Saves the plot to ./assays/plots/

            By default, the full dataframe will be used for the plotting.
            Other dataframes (such as from other assays) can also be used.

            parameters:
            ---------------
            scan_number - float corresponding to the scan of interest
            dataframe - optional arg; pandas dataframe with Scan/Intensity
        """
        if dataframe is None:
            dataframe = self.data
        slice_df = dataframe.loc[
            (dataframe["Scan"] >= scan_number - 1.5) & (dataframe["Scan"] <= scan_number + 1.5)
            ]
        scan_numbers = np.unique(np.round(slice_df["Scan"]))

        with plt.style.context("publication"):
            fig, ax = plt.subplots()

            ax.plot(slice_df["Scan"], slice_df["Intensity"])

            ax.set_xticks(scan_numbers)
            # Makes the x axis not scientific notation
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            ax.set_xlabel("Scan number")
            ax.set_ylabel("Intensity")

            fig.savefig(
                "./assays/plots/{}.pdf".format(scan_number),
                format="pdf",
                transparent=True
                )


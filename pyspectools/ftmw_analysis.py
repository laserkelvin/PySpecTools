
import datetime
import re
import os
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import List, Dict

import pandas as pd
import numpy as np
import peakutils
from matplotlib import pyplot as plt
from scipy import signal as spsig
import plotly.graph_objs as go

from pyspectools import routines
from pyspectools import fitting
from pyspectools.spectra import analysis


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


@dataclass
class Batch:
    assay: str
    id: int


@dataclass
class Scan:
    """
    DataClass for a Scan. Holds all of the relevant information that
    describes a FT scan, such as the ID, what machine it was collected
    on, and the experimental settings.

    Has a few class methods that will make look ups easily such as
    the date the scan was collected and the gases used.
    """
    id: int
    machine: str
    fid: np.array
    date: datetime.datetime
    shots: int = 0
    cavity_voltage: int = 0
    cavity_atten: int = 0
    cavity_frequency: float = 0.0
    dr_frequency: float = 0.0
    dr_power: int = 0
    fid_points: int = 0
    fid_spacing: float = 0.0
    discharge: bool = False
    magnet: bool = False
    gases: Dict = field(default_factory = dict)
    filter: List = field(default_factory = list)
    exp: float = 0.
    zeropad: bool = False
    window: str = ""


    @classmethod
    def from_dict(cls, data_dict):
        """
        Function to initialize a Scan object from a dictionary
        of FT scan data collected from `parse_scan`.
        :param data_dict: dict containing parsed data from FT
        :return: Scan object
        """
        scan_obj = cls(**data_dict)
        return scan_obj

    @classmethod
    def from_qtftm(cls, filepath):
        """
        Method to initialize a Scan object from a FT scan file.
        Will load the lines into memory and parse the data into
        a dictionary, which then gets passed into a Scan object.
        :param filepath: str path to FID file
        :return: Scan object
        """
        with open(filepath) as read_file:
            data_dict = parse_scan(read_file.readlines())
        scan_obj = cls(**data_dict)
        return scan_obj

    @classmethod
    def from_pickle(cls, filepath):
        """
        Method to create a Scan object from a previously pickled
        Scan.
        :param filepath: path to the Scan pickle
        :return: instance of the Scan object
        """
        scan_dict = routines.read_obj(filepath)
        return scan_dict

    def to_file(self, filepath, format="yaml"):
        """ Method to dump data to YAML format.
            Extensions are automatically decided, but
            can also be supplied.

            parameters:
            --------------------
            :param filepath - str path to yaml file
            :param format - str denoting the syntax used for dumping. Defaults to YAML.
        """
        if "." not in filepath:
            if format == "json":
                filepath+=".json"
            else:
                filepath+=".yml"
        if format == "json":
            writer = routines.dump_json
        else:
            writer = routines.dump_yaml
        writer(filepath, self.__dict__)

    def to_pickle(self, filepath=None, **kwargs):
        """
        Pickles the Scan object with the joblib wrapper implemented
        in routines.
        :param filepath: optional argument to pickle to. Defaults to the id.pkl
        :param kwargs: additional settings for the pickle operation
        """
        if filepath is None:
            filepath = "{}.pkl".format(self.id)
        routines.save_obj(self, filepath, **kwargs)

    def process_fid(self, **kwargs):
        """
        Perform an FFT on the FID to yield the frequency domain spectrum.
        Kwargs are passed into the FID processing, which will override the
        Scan attributes.
        :param kwargs: Optional keyword arguments for processing the FID
        """
        # Calculate the frequency bins
        frequencies = np.linspace(
            self.cavity_frequency,
            self.cavity_frequency + 1.,
            len(self.fid)
        )
        # Calculate the time bins
        time = np.linspace(
            0.,
            self.fid_spacing * self.fid_points,
            self.fid_points
        )
        process_list = ["window", "filter", "exp", "zeropad"]
        process_dict = {key: value for key, value in self.__dict__.items() if key in process_list}
        # Override with user settings
        process_dict.update(**kwargs)
        self.spectrum = fid2fft(self.fid, 1. / self.fid_spacing, frequencies, **process_dict)
        self.fid_df = pd.DataFrame({"Time (us)": time * 1e6, "FID": self.fid})


def parse_scan(filecontents):
    """
    Function for extracting the FID data from an FT scan. The data
    is returned as a dictionary, which can be used to initialize a
    Scan object.
    :param filecontents: list of lines from an FID file
    :return: dict containing parsed data from FID
    """
    data = {"gases": dict()}
    # FID regex
    fid_regex = re.compile(r"^fid\d*", re.M)
    # Regex to find gas channels
    gas_regex = re.compile(r"^#Gas \d name", re.M)
    flow_regex = re.compile(r"^#Gas \d flow", re.M)
    # Regex to detect which channel is set to the discharge
    dc_regex = re.compile(r"^#Pulse ch \d name\s*DC", re.M)
    dc_channel = None
    for index, line in enumerate(filecontents):
        if "#Scan" in line:
            split_line = line.split()
            data["id"] = int(split_line[1])
            data["machine"] = split_line[2]
        if "#Probe freq" in line:
            data["cavity_frequency"] = float(line.split()[2])
        if "#Shots" in line:
            data["shots"] = int(line.split()[-1])
        if "#Date" in line:
            strip_targets = ["#Date", "\t", "\n"]
            data["date"] = datetime.datetime.strptime(
                re.sub("|".join(strip_targets), "", line),
                "%a %B %d %H:%M:%S %Y"
            )
        if "#Cavity Voltage" in line:
            data["cavity_voltage"] = int(line.split()[2])
        if "#Attenuation" in line:
            data["cavity_atten"] = int(line.split()[1])
        if "#DR freq" in line:
            data["dr_frequency"] = float(line.split()[2])
        if "#DR power" in line:
            data["dr_power"] = int(line.split()[2])
        if "#FID spacing" in line:
            data["fid_spacing"] = float(
                    re.findall(
                    r"\de[+-]?\d\d",
                    line
                )[0]
            )
        if "#FID points" in line:
            data["fid_points"] = int(line.split()[-1])
        # Get the name of the gas
        if gas_regex.match(line):
            split_line = line.split()
            # Only bother parsing if the channel is used
            gas_index = int(split_line[1])
            try:
                data["gases"][gas_index] = {"gas": " ".join(split_line[3:])}
            except IndexError:
                data["gases"][gas_index] = {"gas": ""}
        # Get the flow rate for channel
        if flow_regex.match(line):
            split_line = line.split()
            gas_index = int(split_line[1])
            data["gases"][gas_index]["flow"] = float(split_line[3])
        if "#Magnet enabled" in line:
            data["magnet"] = bool(int(line.split()[2]))
        # Find the channel the discharge is set to and compile a regex
        # to look for the channel
        if dc_regex.match(line):
            dc_index = line.split()[2]
            dc_channel = re.compile(r"^#Pulse ch {} enabled".format(dc_index), re.M)
        # Once the discharge channel index is known, start searching for it
        if dc_channel:
            if dc_channel.match(line):
                data["discharge"] = bool(int(line.split()[-1]))
        # Find when the FID lines start popping up
        if fid_regex.match(line):
            fid = filecontents[index+1:]
            fid = [float(value) for value in fid]
            data["fid"] = np.array(fid)
    return data


def fid2fft(fid, rate, frequencies, **kwargs):
    """
    Process an FID by performing an FFT to yield the frequency domain
    information. Kwargs are passed as additional processing options,
    and are implemented as some case statements to ensure the settings
    are valid (e.g. conforms to sampling rate, etc.)

    :param fid: np.array corresponding to the FID intensity
    :param rate: sampling rate in Hz
    :param frequencies: np.array corresponding to the frequency bins
    :param kwargs: signal processing options:
                    delay - delays the FID processing by setting the start
                            of the FID to zero
                    zeropad - Toggles whether or not the number of sampled
                              points is doubled to get artificially higher
                              resolution in the FFT
                    window - Various window functions provided by `scipy.signal`
                    exp - Specifies an exponential filter
                    filter - 2-tuple specifying the frequency cutoffs for a
                             band pass filter
    :return: freq_df - pandas dataframe with the FFT spectrum
    """
    # Remove DC
    fid-=np.average(fid)
    if "delay" in kwargs:
        if 0 < kwargs["delay"] < len(fid) * (1e6 / rate):
            fid[:int(delay)] = 0.
    # Zero-pad the FID
    if "zeropad" in kwargs:
        if kwargs["zeropad"] is True:
            # Pad the FID with zeros to get higher resolution
            fid = np.append(fid, np.zeros(len(fid)))
            # Since we've padded with zeros, we'll have to update the
            # frequency array
            frequencies = spsig.resample(frequencies, len(frequencies) * 2)
    # Apply a window function to the FID
    if "window" in kwargs:
        available = ["blackmanharris", "blackman", "boxcar", "gaussian", "hanning", "bartlett"]
        if (kwargs["window"] != "") and (kwargs["window"] in available):
            fid*=spsig.get_window(kwargs["window"], len(fid))
    # Apply an exponential filter on the FID
    if "exp" in kwargs:
        if kwargs["exp"] > 0.:
            fid*=spsig.exponential(len(fid), tau=kwargs["exp"])
    # Apply a bandpass filter on the FID
    if ("filter" in kwargs) and (len(kwargs["filter"]) == 2):
        low, high = sorted(kwargs["filter"])
        if (low < high) and (high <= 950.):
            fid = apply_butter_filter(
                fid,
                low,
                high,
                1. / rate
            )
    # Perform the FFT
    fft = np.fft.fft(fid)
    # Get the real part of the FFT
    real_fft = np.abs(fft[:int(len(fid) / 2)]) / len(fid)
    frequencies = spsig.resample(frequencies, len(real_fft))
    # For some reason, resampling screws up the frequency ordering...
    frequencies = np.sort(frequencies)
    # Package into a pandas dataframe
    freq_df = pd.DataFrame({"Frequency (MHz)": frequencies, "Intensity": real_fft})
    return freq_df


def butter_bandpass(low, high, rate, order=5):
    """
        A modified version of the Butterworth bandpass filter described here,
        adapted for use with the FID signal.
        http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        The arguments are:

        :param low The low frequency cut-off, given in kHz.
        :param high The high frequency cut-off, given in kHz.
        :param rate The sampling rate, given in Hz. From the FIDs, this means that
                    the inverse of the FID spacing is used.
        :return bandpass window
    """
    # Calculate the Nyquist freqiemcy
    nyq = 0.5 * (rate / (2. * np.pi))
    low = (low * 1e3) / nyq
    high = (high * 1e3) / nyq
    b, a = spsig.butter(order, [low, high], btype='band', analog=False)
    return b, a


def apply_butter_filter(data, low, high, rate, order=5):
    """
        A modified Butterworth bandpass filter, adapted from the Scipy cookbook.

        The argument data supplies the FID, which then uses the scipy signal
        processing function to apply the digital filter, and returns the filtered
        FID.

        See the `butter_bandpass` function for additional arguments.
    """
    b, a = butter_bandpass(low, high, rate, order=order)
    y = spsig.lfilter(b, a, data)
    return y


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
        cal

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
        """
            Create an AssayBatch session by providing a
            filepath to a file containing the frequency and
            intensity information, as well as the experiment
            ID from which it was based.

            parameters:
            --------------
            filepath - path to a CSV file with Frequency/Intensity
                       columns
            exp_id - experiment ID; typically the chirp experiment number
        """
        df = pd.read_csv(filepath)
        return cls(df, exp_id)

    @classmethod
    def load_session(cls, filepath):
        """
            Loads a previously saved AssayBatch session.
        """
        session = routines.read_obj(filepath)
        obj = cls(**session)
        return obj

    def __init__(self, data, exp_id, freq_col="Frequency", int_col="Intensity"):
        folders = ["assays", "ftbfiles"]
        for folder in folders:
            if os.path.isdir(folder) is False:
                os.mkdir(folder)

        if os.path.isdir("assays/plots") is False:
            os.mkdir("./assays/plots")

        self.data = data
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
            Generate an FT batch file for performing magnetic tests.

            parameters:
            -----------------
            dataframe - dataframe used for the frequency information.
                        By default the dipole dataframe will be used.
            cal - bool denoting whether a calibration line is used.
                  A user can provide it in kwargs, or use the default
                  which is the first line in the dataframe (strongest
                  if sorted by intensity)
            nshots - optional int specifying number of shots for each
                     line, or if SNR is in the dataframe, the number
                     of shots for the strongest line.
            kwargs - passed into the calibration settings, which are
                     cal_freq, cal_rate, and cal_shots
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

        # If calibration is requested
        # cal_rate sets the number of scans per calibration
        if cal is True:
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

    def generate_bruteforce_dr(self, nshots=10, dr_channel=3):
        """
            Brute force double resonance test on every single frequency
            observed in the initial dipole test.

            This method will perform DR measurements on sequentially
            weaker lines, if the dipole_df is sorted by SNR.

            Optionally, the 
        """
        ftb_str = ""
        combos = combinations(self.dipole_df[["Frequency", "Dipole"]].values, 2)
        for index, combo in enumerate(combos):
            # For the first time a cavity frequency is used
            # we will measure it once without DR
            freq = combo[0][0]
            if (index == 0) or (last_freq != freq):
                ftb_str+=generate_ftb_line(
                    freq,
                    nshots,
                    **{
                        "dipole": combo[0][1],
                        "pulse,{},enable".format(dr_channel): "false",
                        "skiptune": "false",
                        }
                    )
            # Only do DR if the DR frequency is significantly different from
            # the cavity frequency
            if np.abs(combo[1][0] - freq) >= 100.:
                ftb_str+=generate_ftb_line(
                    freq,
                    nshots,
                    **{
                        "dipole": combo[0][1],
                        "drfreq": combo[1][0],
                        "pulse,{},enable".format(dr_channel): "true",
                        "skiptune": "true"
                        }
                    )
            last_freq = combo[0][0]
        print("There are {} combinations to measure.".format(index))
        
        with open("./ftbfiles/{}-bruteDR.ftb".format(self.exp_id), "w+") as write_file:
            write_file.write(ftb_str)

        print("FTB file saved to ./ftbfiles/{}-bruteDR.ftb".format(self.exp_id))

    def find_progressions(self, **kwargs):
        """
            Uses the dipole assay data to look for harmonic
            progression.

            Kwargs are passed into the affinity propagation
            clustering; usually this means "preference" should
            be set to tune the number of clusters.

            returns:
            --------------
            cluster_dict - dictionary containing all of the clustered
                           progressions
        """
        progressions = analysis.harmonic_finder(
            self.dipole_df["Frequency"].values
            )
        self.progression_df = fitting.harmonic_fitter(progressions)
        data, ap_obj = analysis.cluster_AP_analysis(
            self.progression_df,
            True,
            False,
            **kwargs
            )
        self.cluster_dict = data
        self.cluster_obj = ap_obj
        return self.cluster_dict

    def generate_progression_test(self, nshots=50, dr_channel=3):
        """
            Take the dipole moment dataframe and generate a DR batch.
            If possible, we will instead use the progressions predicted
            by the cluster model.
        """
        ftb_str = ""
        count = 0
        # Loop over progressions
        for index, sub_dict in self.cluster_dict.items():
            # Take only frequencies that are in the current progression
            slice_df = self.dipole_df.loc[
                self.dipole_df["Frequency"].isin(sub_dict["Frequencies"])
                ]
            prog_data = slice_df[["Frequency", "Dipole", "Shots"]].values
            for sub_index, pair in enumerate(combinations(prog_data, 2)):
                count+=1
                if (sub_index == 0) or (last_freq != pair[0][0]):
                    ftb_str+=generate_ftb_line(
                        pair[0][0],
                        10,
                        **{
                            "dipole": pair[0][1],
                            "pulse,{},enable".format(dr_channel): "false",
                            "skiptune": "false"
                            }
                        )
                # Perform the DR measurement
                ftb_str+=generate_ftb_line(
                    pair[0][0],
                    10,
                    **{
                        "dipole": pair[0][1],
                        "pulse,{},enable".format(dr_channel): "true",
                        "skiptune": "true"
                        }
                    )
                last_freq = pair[0][0]
        print("There are {} combinations to test.".format(count))

        with open("./ftbfiles/{}-progressionDR.ftb".format(self.exp_id), "w+") as write_file:
            write_file.write(ftb_str)

        print("FTB file saved to ./ftbfiles/{}-progressionDR.ftb".format(self.exp_id))

    def plot_scan(self, scan_number=None):
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
        
        if scan_number is not None:
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

    def save_session(self, filepath=None):
        """
            Method to save the current assay analysis session to disk.

            The data can be reloaded using the AssayBatch.load_session
            class method.

            parameters:
            ---------------
            filepath - path to save the data to. By default, the path will
                       be the experiment ID.
        """
        if filepath is None:
            filepath = "./assays/{}-assay-analysis.dat".format(self.exp_id)
        routines.save_obj(
            self.__dict__,
            filepath
            )
        print("Saved session to {}".format(filepath))

import datetime
import re
import os
import struct
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import List, Dict

import pandas as pd
import numpy as np
import peakutils
from matplotlib import pyplot as plt
from scipy import signal as spsig
import plotly.graph_objs as go
from tqdm.autonotebook import tqdm
import networkx as nx
from ipywidgets import interactive, VBox, HBox
from lmfit.models import LinearModel

from pyspectools import routines
from pyspectools import figurefactory as ff
from pyspectools import fitting
from pyspectools.spectra import analysis
from pyspectools import parsers


def parse_specdata(filename):
    # For reading the output of a SPECData analysis
    return pd.read_csv(filename, skiprows=4)


def parse_spectrum(filename, threshold=20.0):
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
    machine: str
    date: datetime.datetime
    scans: List = field(default_factory=list)
    filter: List = field(default_factory=list)
    exp: float = 0.0
    zeropad: bool = False
    window: str = ""

    @classmethod
    def from_qtftm(cls, filepath, assay, machine):
        """
        Create a Batch object from a QtFTM scan file.
        :param filepath:
        :param assay:
        :param machine:
        :return:
        """
        assays = ["dr", "magnet", "discharge", "dipole"]
        assay = assay.lower()
        if assay not in assays:
            raise Exception(
                "Not a valid assay type; choose dr, magnet, discharge, dipole."
            )
        with open(filepath) as read_file:
            batch_df, batch_data = parse_batch(read_file.readlines())
        batch_data["assay"] = assay
        batch_data["machine"] = machine.upper()
        batch_obj = cls(**batch_data)
        batch_obj.details = batch_df
        return batch_obj

    @classmethod
    def from_remote(cls, root_path, batch_id, assay, machine, ssh_obj=None):
        """
        Create a Batch object by retrieving a QtFTM batch file from a remote
        location. The user must provide a path to the root directory of the
        batch type, e.g. /home/data/QtFTM/batch/, and the corresponding
        batch type ID. Additionally, the type of batch must be specified to
        determine the type of analysis required.

        Optionally, the user can provide a reference to a RemoteClient object
        from `pyspectools.routines`. If none is provided, a RemoteClient object
        will be created with public key authentication (if available), otherwise
        the user will be queried for a hostname, username, and password.

        Keep in mind this method can be slow as every scan is downloaded. I
        recommend creating this once, then saving a local copy for subsequent
        analysis.
        :param root_path: str path to the batch type root directory
        :param batch_id: int or str value for the batch number
        :param assay: str the batch type
        :param machine: str reference the machine used for the batch
        :param ssh_obj: `RemoteClient` object
        :return:
        """
        if ssh_obj is None:
            default_keypath = os.path.join(os.path.expanduser("~"), ".ssh/id_rsa.pub")
            hostname = input("Please provide remote hostname:    ")
            username = input("Please provide login:              ")
            ssh_settings = {"hostname": hostname, "username": username}
            if os.path.isfile(default_keypath) is True:
                ssh_settings["key_filename"] = default_keypath
            else:
                password = input("Please provide password:              ")
                ssh_settings["password"] = password
            ssh_obj = routines.RemoteClient(**ssh_settings)
        # Parse the scan data from remote file
        remote_path = ssh_obj.ls(
            os.path.join(root_path, "*", "*", str(batch_id) + ".txt")
        )
        batch_df, batch_data = parse_batch(ssh_obj.open_remote(remote_path[0]))
        batch_data["assay"] = assay
        batch_data["machine"] = machine.upper()
        batch_obj = cls(**batch_data)
        batch_obj.details = batch_df
        batch_obj.remote = ssh_obj
        batch_obj.get_scans(root_path, batch_df.id.values)
        return batch_obj

    @classmethod
    def from_pickle(cls, filepath):
        """
        Method to create a Scan object from a previously pickled
        Scan.
        :param filepath: path to the Scan pickle
        :return: instance of the Scan object
        """
        batch_obj = routines.read_obj(filepath)
        if isinstance(batch_obj, Batch) is False:
            raise Exception("File is not a Scan object; {}".format(type(batch_obj)))
        else:
            return batch_obj

    def __repr__(self):
        return "{}-Batch {}".format(self.machine, self.id)

    def __copy__(self):
        batch_obj = Batch(**self.__dict__)
        return batch_obj

    def find_scan(self, id):
        scans = [scan for scan in self.scans if scan.id == id]
        if len(scans) == 0:
            raise Exception("No scans were found.")
        else:
            return scans[0]

    def get_scans(self, root_path, ids):
        """
        Function to create Scan objects for all of the scans in
        a QtFTM batch.
        :param root_path: str scans root path
        :param ids: list scan ids
        :param src: str optional specifying whether a remote or local path is used
        """
        root_path = root_path.replace("batch", "scans")
        path_list = tqdm(
            [
                os.path.join(root_path, "*", "*", str(scan_id) + ".txt")
                for scan_id in ids
            ]
        )
        if hasattr(self, "remote") is True:
            scans = [Scan.from_remote(path, self.remote) for path in path_list]
        else:
            scans = [Scan.from_qtftm(path) for path in path_list]
        self.scans = scans

    def process_dr(self, significance=16.0):
        """
        Function to batch process all of the DR measurements.
        :param global_depletion: float between 0. and 1. specifying the expected depletion for any line
                                 without a specific expected value.
        :param depletion_dict: dict with keys corresponding to cavity frequencies, and values the expected
                               depletion value between 0. and 1.
        :return dr_dict: dict with keys corresponding to cavity frequency, with each value
                         a dict of the DR frequencies, scan IDs and Scan objects.
        """
        if self.assay != "dr":
            raise Exception(
                "Batch is not a DR test! I think it's {}".format(self.assay)
            )
        # Find the cavity frequencies that DR was performed on
        progressions = self.split_progression_batch()
        dr_dict = dict()
        counter = 0
        for index, progression in tqdm(progressions.items()):
            ref = progression.pop(0)
            try:
                ref_fit = ref.fit_cavity(plot=False)
                roi, ref_x, ref_y = ref.get_line_roi()
                signal = [np.sum(ref_y, axis=0)]
                sigma = (
                    np.average(
                        [
                            np.std(scan.spectrum["Intensity"].iloc[roi])
                            for scan in progression
                        ]
                    )
                    * significance
                )
                connections = [
                    scan for scan in progression if scan.is_depleted(ref, roi, sigma)
                ]
                if len(connections) > 1:
                    counter += len(connections)
                    signal.extend(
                        [
                            np.sum(scan.spectrum["Intensity"].iloc[roi], axis=0)
                            for scan in connections
                        ]
                    )
                    dr_dict[index] = {
                        "frequencies": [scan.dr_frequency for scan in connections],
                        "ids": [scan.id for scan in connections],
                        "cavity": ref.fit.frequency,
                        "signal": signal,
                        "expected": np.sum(ref_y) - sigma,
                    }
            except ValueError:
                print("Progression {} could not be fit; ignoring.".format(index))
        print(
            "Possible depletions detected in these indexes: {}".format(
                list(dr_dict.keys())
            )
        )
        print("There are {} possible depletions.".format(counter))
        return dr_dict

    def split_progression_batch(self):
        """
        Split up a DR batch into individual progressions based on the cavity frequency
        and whether or not the scan IDs are consecutive.
        :return progressions: dict with keys corresponding to progression index and values are lists of Scans
        """
        counter = 0
        progressions = dict()
        self.details["id"] = self.details["id"].apply(int)
        for freq in self.details["ftfreq"].unique():
            slice_df = self.details.loc[self.details["ftfreq"] == freq]
            chunks = routines.group_consecutives(slice_df["id"])
            for chunk in chunks:
                progressions[counter] = [
                    scan for scan in self.scans if scan.id in chunk
                ]
                counter += 1
        return progressions

    def interactive_dr_batch(self):
        """
        Create an interactive widget slider with a Plotly figure. The batch will be split
        up into "subbatches" by the cavity frequency and whether or not the scan IDs are
        consecutive.
        :return vbox: VBox object with the Plotly figure and slider objects
        """
        progressions = self.split_progression_batch()
        fig = go.FigureWidget()
        fig.layout["width"] = 900.0
        fig.layout["showlegend"] = False

        def update_figure(index):
            fig.data = []
            fig.add_traces([scan.scatter_trace() for scan in progressions[index]])

        index_slider = interactive(update_figure, index=(0, len(progressions) - 1, 1))
        vbox = VBox((fig, index_slider))
        vbox.layout.align_items = "center"
        return vbox

    def plot_scans(self):
        """
        Create a plotly figure of all of the Scans within a Batch.
        :return:
        """
        fig = go.FigureWidget()
        fig.layout["title"] = "{} Batch {}".format(self.machine, self.id)
        fig.layout["showlegend"] = False
        fig.add_traces([scan.scatter_trace() for scan in self.scans])
        return fig

    def reprocess_fft(self, **kwargs):
        """
        Reprocess all of the FIDs with specified settings. The default values
        are taken from the Batch attributes, and kwargs provided will override
        the defaults.
        :param kwargs:
        """
        param_list = ["filter", "exp", "zeropad", "window"]
        params = {
            key: value for key, value in self.__dict__.items() if key in param_list
        }
        params.update(**kwargs)
        _ = [scan.process_fid(**params) for scan in tqdm(self.scans)]

    def to_pickle(self, filepath=None, **kwargs):
        """
        Pickles the Batch object with the joblib wrapper implemented
        in routines.
        :param filepath: optional argument to pickle to. Defaults to the {assay}-{id}.pkl
        :param kwargs: additional settings for the pickle operation
        """
        if filepath is None:
            filepath = "{}-{}.pkl".format(self.assay, self.id)
        # the RemoteClient object has some thread locking going on that prevents
        # pickling TODO - figure out why paramiko doesn't allow pickling
        if hasattr(self, "remote"):
            delattr(self, "remote")
        routines.save_obj(self, filepath, **kwargs)

    def create_dr_network(self, scans):
        """
        Take a list of scans, and generate a NetworkX Graph object
        for analysis and plotting.
        :param scans: list of scan IDs to connect
        :return fig: Plotly FigureWidget object
        """
        connections = [
            [np.floor(scan.cavity_frequency), np.floor(scan.dr_frequency)]
            for scan in self.scans
            if scan.id in scans
        ]
        fig, self.progressions = ff.dr_network_diagram(connections)
        return fig

    def find_optimum_scans(self, thres=0.8):
        """

        :param thres:
        :return:
        """
        progressions = self.split_progression_batch()
        data = list()
        for index, progression in tqdm(progressions.items()):
            snrs = [scan.calc_snr(thres=thres) for scan in progression]
            best_scan = progression[np.argmax(snrs)]
            try:
                fit_result = best_scan.fit_cavity(plot=False)
                if fit_result.best_values["w"] < 0.049:
                    data.append(
                        {
                            "frequency": np.round(fit_result.best_values["x0"], 4),
                            "snr": np.max(snrs),
                            "scan": best_scan.id,
                            "attenuation": best_scan.cavity_atten,
                            "index": index,
                        }
                    )
            except ValueError:
                print("Index {} failed to fit!".format(index))
        opt_df = pd.DataFrame(data)
        return opt_df

    def search_frequency(self, frequency, tol=0.001):
        """
        Search the Batch scans for a particular frequency, and return
        any scans that lie within the tolerance window
        :param frequency: float specifying frequency to search
        :param tol: float decimal percentage to use for the search tolerance
        :return new_batch: a new Batch object with selected scans
        """
        upper = frequency * (1 + tol)
        lower = frequency * (1 - tol)
        scans = [scan for scan in self.scans if lower <= scan.cavity_frequency <= upper]
        # new_batch = deepcopy(self)
        # new_batch.scans = scans
        return scans


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
    gases: Dict = field(default_factory=dict)
    filter: List = field(default_factory=list)
    exp: float = 0.0
    zeropad: bool = False
    window: str = ""

    def __post_init__(self):
        """
        Functions called after __init__ is called.
        """
        # Perform FFT
        self.process_fid()

    def __deepcopy__(self):
        """
        Dunder method to produce a deep copy - this will be used when
        manipulating multiple Scan objects.
        :return: A deep copy of the current Scan object
        """

        class Empty(self.__class__):
            def __init__(self):
                pass

        new_scan = Empty()
        new_scan.__class__ = self.__class__
        new_scan.__dict__.update(self.__dict__)
        return new_scan

    def __repr__(self):
        return str(f"Scan {self.id}")

    def average(self, others):
        """
        Dunder method to co-average two or more Scans in the time domain.
        :param other: Scan object, or tuple/list
        :return: A new Scan object with the co-added FID
        """
        new_scan = self.__deepcopy__()
        try:
            new_scan.fid = np.average(others.extend(new_scan.fid), axis=0)
            new_scan.average_ids = [scan.id for scan in others]
        # If there is no extend method, then assume we're working with a
        # single Scan
        except AttributeError:
            new_scan.fid = np.average([new_scan.fid, others.fid], axis=0)
            new_scan.average_ids = [others.id]
        new_scan.process_fid()
        return new_scan

    def __add__(self, other):
        """
        Dunder method to co-add two or more Scans in the time domain.
        :param other: Scan object, or tuple/list
        :return: A new Scan object with the co-added FID
        """
        new_scan = self.__deepcopy__()
        new_scan.fid = np.sum([new_scan.fid, other.fid], axis=0)
        new_scan.process_fid()
        return new_scan

    def __sub__(self, other):
        """
        Dunder method to subtract another Scan from the current Scan in the time domain.
        i.e. this scan - other scan
        :param other: Scan object, or tuple/list
        :return: A new Scan object with the subtracted FID
        """
        new_scan = self.__deepcopy__()
        new_scan.fid = np.subtract(new_scan.fid, other.fid)
        new_scan.process_fid()
        return new_scan

    def subtract_frequency(self, other):
        """
        Method to subtract another Scan from the current in the frequency domain.
        :param other: Scan object to subtract with
        :return: A new Scan object with the subtracted spectrum
        """
        new_scan = self.__deepcopy__()
        new_scan.spectrum["Intensity"] = (
            new_scan.spectrum["Intensity"] - other.spectrum["Intensity"]
        )
        new_scan.subtracted = other.id
        return new_scan

    def add_frequency(self, other):
        """
        Method to add another Scan from the current in the frequency domain.
        :param other: Scan object to add with
        :return: A new Scan object with the co-added spectrum
        """
        new_scan = self.__deepcopy__()
        new_scan.spectrum["Intensity"] = (
            new_scan.spectrum["Intensity"] + other.spectrum["Intensity"]
        )
        new_scan.subtracted = other.id
        return new_scan

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
        scan_obj = routines.read_obj(filepath)
        if isinstance(scan_obj, Scan) is False:
            raise Exception("File is not a Scan object; {}".format(type(scan_obj)))
        else:
            return scan_obj

    @classmethod
    def from_remote(cls, remote_path, ssh_obj=None):
        """
        Method to initialize a Scan object from a remote server.
        Has the option to pass an instance of a paramiko SSHClient, which would be
        useful in a Batch. If none is supplied, an instance will be created.

        :param remote_path: str remote path to the file
        :param ssh_obj: optional argument to supply a paramiko SSHClient object
        :return: Scan object from remote QtFTM file
        """
        if ssh_obj is None:
            default_keypath = os.path.join(os.path.expanduser("~"), ".ssh/id_rsa.pub")
            hostname = input("Please provide remote hostname:    ")
            username = input("Please provide login:              ")
            ssh_settings = {"hostname": hostname, "username": username}
            if os.path.isfile(default_keypath) is True:
                ssh_settings["key_filename"] = default_keypath
            else:
                password = input("Please provide password:              ")
                ssh_settings["password"] = password
            ssh_obj = routines.RemoteClient(**ssh_settings)
        # Parse the scan data from remote file
        data_dict = parse_scan(ssh_obj.open_remote(remote_path))
        scan_obj = cls(**data_dict)
        return scan_obj

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
                filepath += ".json"
            else:
                filepath += ".yml"
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
            self.cavity_frequency, self.cavity_frequency + 1.0, len(self.fid)
        )
        # Calculate the time bins
        time = np.linspace(0.0, self.fid_spacing * self.fid_points, self.fid_points)
        process_list = ["window", "filter", "exp", "zeropad"]
        process_dict = {
            key: value for key, value in self.__dict__.items() if key in process_list
        }
        # Override with user settings
        process_dict.update(**kwargs)
        temp_fid = np.copy(self.fid)
        self.spectrum = fid2fft(
            temp_fid, 1.0 / self.fid_spacing, frequencies, **process_dict
        )
        self.fid_df = pd.DataFrame({"Time (us)": time * 1e6, "FID": temp_fid})

    def within_time(self, date_range):
        """
        Function for determining of the scan was taken between
        a specified date range in month/day/year, in the format
        04/09/08 for April 9th, 2008.
        :param date_range: list containing the beginning and end date strings
        :return: bool - True if within range, False otherwise
        """
        try:
            early = datetime.datetime.strptime(date_range[0], "%m/%d/%y")
        except:
            early = datetime.datetime(1, 1, 1)
        try:
            late = datetime.datetime.strptime(date_range[1], "%m/%d/%y")
        except:
            late = datetime.datetime(9999, 1, 1)
        return early <= self.date <= late

    def is_depleted(self, ref, roi=None, depletion=None):
        """
        Function for determining if the signal in this Scan is less
        than that of another scan. This is done by a simple comparison
        of the average of 10 largest intensities in the two spectra. If
        the current scan is less intense than the reference by the
        expected depletion percentage, then it is "depleted".

        This function can be used to determine if a scan if depleted
        in DR/magnet/discharge assays.

        TODO - implement a chi squared test of sorts to determine if a
               depletion is statistically significant

        :param ref: second Scan object for comparison
        :param depletion: percentage of depletion expected of the reference
        :return: bool - True if signal in this Scan is less intense than the reference
        """
        y_ref = ref.spectrum["Intensity"].values
        y_obs = self.spectrum["Intensity"].values
        self.ref_freq = ref.fit.frequency
        self.ref_id = ref.id
        if roi:
            y_ref = y_ref[roi]
            y_obs = y_obs[roi]
        # This doesn't work, or is not particularly discriminating.
        # chisq, p_value = chisquare(
        #    y_obs, y_ref
        # )
        if depletion is None:
            sigma = np.std(y_obs, axis=0) * 16.0
        else:
            sigma = depletion
        expected = np.sum(y_ref, axis=0) - sigma
        return np.sum(y_obs, axis=0) <= expected

    def scatter_trace(self):
        """
        Create a Plotly Scattergl trace. Called by the Batch function, although
        performance-wise it takes forever to plot up ~3000 scans.
        :return trace: Scattergl object
        """
        text = "Scan ID: {}<br>Cavity: {}<br>DR: {}<br>Magnet: {}<br>Attn: {}".format(
            self.id,
            self.cavity_frequency,
            self.dr_frequency,
            self.magnet,
            self.cavity_atten,
        )
        trace = go.Scattergl(
            x=np.linspace(self.id, self.id + 1, len(self.spectrum["Intensity"])),
            y=self.spectrum["Intensity"],
            text=text,
            marker={"color": "rgb(43,140,190)"},
            hoverinfo="text",
        )
        return trace

    def fit_cavity(self, plot=True, verbose=False):
        """
        Perform a fit to the cavity spectrum. Uses a paired Gaussian model
        that minimizes the number of fitting parameters.
        :param plot: bool specify whether a Plotly figure is made
        :return: Model Fit result
        """
        y = self.spectrum["Intensity"].dropna().values
        x = self.spectrum["Frequency (MHz)"].dropna().values
        model = fitting.PairGaussianModel()
        result = model.fit_pair(x, y, verbose=verbose)
        self.spectrum["Fit"] = result.best_fit
        self.fit = result
        self.fit.frequency = self.fit.best_values["x0"]
        if plot is True:
            fig = go.FigureWidget()
            fig.layout["xaxis"]["title"] = "Frequency (MHz)"
            fig.layout["xaxis"]["tickformat"] = ".2f"
            fig.add_scatter(x=x, y=y, name="Observed")
            fig.add_scatter(x=x, y=result.best_fit, name="Fit")
            return result, fig
        else:
            return result

    def get_line_roi(self):
        if hasattr(self, "fit") is False:
            raise Exception("Auto peak fitting has not been run yet!")
        # Get one of the Doppler horns plus 4sigma
        params = self.fit.best_values
        x = self.spectrum["Frequency (MHz)"].values
        y = self.spectrum["Intensity"].values
        _, low_end = routines.find_nearest(
            x, params["x0"] - params["xsep"] - params["w"] * 4.0
        )
        _, high_end = routines.find_nearest(
            x, params["x0"] + params["xsep"] + params["w"] * 4.0
        )
        index = list(range(low_end, high_end))
        return index, x[low_end:high_end], y[low_end:high_end]

    def calc_snr(self, noise=None, thres=0.6):
        if noise is None:
            # Get the last 10 points at the end and at the beginning
            noise = np.average(
                [
                    self.spectrum["Intensity"].iloc[-10:],
                    self.spectrum["Intensity"].iloc[:10],
                ]
            )
        peaks = (
            self.spectrum["Intensity"]
            .iloc[
                peakutils.indexes(
                    self.spectrum["Intensity"], thres=thres, thres_abs=True
                )
            ]
            .values
        )
        signal = np.average(np.sort(peaks)[:2])
        return signal / noise


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
            try:
                data["machine"] = split_line[2]
            except IndexError:
                data["machine"] = "FT1"
        if "#Probe freq" in line:
            data["cavity_frequency"] = float(line.split()[2])
        if "#Shots" in line:
            data["shots"] = int(line.split()[-1])
        if "#Date" in line:
            strip_targets = ["#Date", "\t", "\n"]
            data["date"] = datetime.datetime.strptime(
                re.sub("|".join(strip_targets), "", line), "%a %b %d %H:%M:%S %Y"
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
            data["fid_spacing"] = float(re.findall(r"\de[+-]?\d\d", line)[0])
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
            fid = filecontents[index + 1 :]
            fid = [float(value) for value in fid]
            data["fid"] = np.array(fid)
    return data


def perform_fft(fid, spacing, start=0, stop=-1, window="boxcar"):
    """
    Perform an FFT on an FID to get the frequency domain spectrum.
    All of the arguments are optional, and provide control over how the FFT is performed, as well as post-processing
    parameters like window functions and zero-padding.

    This is based on the FFT code by Kyle Crabtree, with modifications to fit this dataclass.

    Parameters
    ----------
    fid - Numpy 1D array
        Array holding the values of the FID
    spacing - float
        Time spacing between FID points in microseconds
    start - int, optional
        Starting index for the FID array to perform the FFT
    stop - int, optional
        End index for the FID array to perform the FFT
    zpf - int, optional
        Pad the FID with zeros to nth nearest power of 2
    window - str
        Specify the window function used to process the FID. Defaults to boxcar, which is effectively no filtering.
        The names of the window functions available can be found at:
        https://docs.scipy.org/doc/scipy/reference/signal.windows.html

    Returns
    -------
    """
    fid = np.copy(fid)
    if window is not None and window in spsig.windows.__all__:
        window_f = spsig.windows.get_window(window, fid.size)
        fid *= window_f
    else:
        raise Exception("Specified window function is not implemented in SciPy!")
    # Set values to zero up to starting index
    fid[:start] = 0.0
    if stop < 0:
        # If we're using negative indexes
        fid[fid.size + stop :] = 0.0
    else:
        # Otherwise, index with a positive number
        fid[stop:] = 0.0
    # Perform the FFT
    fft = np.fft.rfft(fid)
    read_length = len(fid) // 2 + 1
    df = 1.0 / fid.size / spacing
    # Generate the frequency array
    frequency = np.linspace(0.0, self.header["sideband"] * df, read_length)
    frequency += self.header["probe_freq"]
    fft[(frequency >= f_max) & (frequency <= f_min)] = 0.0
    fft *= 1000.0
    return frequency, fft


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
    new_fid = fid - np.average(fid)
    if "delay" in kwargs:
        delay = int(kwargs["delay"] / (1.0 / rate) / 1e6)
        new_fid[:delay] = 0.0
    # Zero-pad the FID
    if "zeropad" in kwargs:
        if kwargs["zeropad"] is True:
            # Pad the FID with zeros to get higher resolution
            fid = np.append(new_fid, np.zeros(len(new_fid)))
            # Since we've padded with zeros, we'll have to update the
            # frequency array
            frequencies = spsig.resample(frequencies, len(frequencies) * 2)
    # Apply a window function to the FID
    if "window" in kwargs:
        if kwargs["window"] in spsig.windows.__all__:
            new_fid *= spsig.get_window(kwargs["window"], new_fid.size)
    # Apply an exponential filter on the FID
    if "exp" in kwargs:
        if kwargs["exp"] > 0.0:
            new_fid *= spsig.exponential(len(new_fid), tau=kwargs["exp"])
    # Apply a bandpass filter on the FID
    if ("filter" in kwargs) and (len(kwargs["filter"]) == 2):
        low, high = sorted(kwargs["filter"])
        if low < high:
            new_fid = apply_butter_filter(new_fid, low, high, rate)
    # Perform the FFT
    fft = np.fft.rfft(new_fid)
    # Get the real part of the FFT, and only the non-duplicated side
    real_fft = np.abs(fft[: int(len(new_fid) / 2)]) / len(new_fid) * 1e3
    frequencies = spsig.resample(frequencies, real_fft.size)
    # For some reason, resampling screws up the frequency ordering...
    real_fft = real_fft[np.argsort(frequencies)]
    frequencies = np.sort(frequencies)
    # Package into a pandas dataframe
    freq_df = pd.DataFrame({"Frequency (MHz)": frequencies, "Intensity": real_fft})
    return freq_df


def butter_bandpass(low, high, rate, order=1):
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
    # Calculate the Nyquist frequency
    nyq = 0.5 * (rate / (2.0 * np.pi))
    low = (low * 1e3) / nyq
    high = (high * 1e3) / nyq
    if high > 1.0:
        raise Exception("High frequency cut-off exceeds the Nyquist frequency.")
    b, a = spsig.butter(order, [low, high], btype="band", analog=False)
    return b, a


def apply_butter_filter(data, low, high, rate, order=1):
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


def parse_batch(filecontents):
    data = dict()
    for index, line in enumerate(filecontents):
        if "#Batch scan" in line:
            data["id"] = int(line.split()[2])
        if "#Date" in line:
            strip_targets = ["#Date", "\t", "\n"]
            data["date"] = datetime.datetime.strptime(
                re.sub("|".join(strip_targets), "", line), "%a %b %d %H:%M:%S %Y"
            )
        if line.startswith("batchscan"):
            scan_details = filecontents[index + 1 :]
            scan_details = [scan.split() for scan in scan_details]
    headers = [
        "id",
        "max",
        "iscal",
        "issat",
        "ftfreq",
        "attn",
        "drfreq",
        "drpower",
        "pulses",
        "shots",
        "autofitpair_freq",
        "autofitpair_int",
        "autofitfreq",
        "autofitint",
    ]
    df = pd.DataFrame(scan_details, columns=headers)
    return df, data


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
        :param frequency: float for frequency in MHz
        :param shots: int number of shots to integrate for

        returns:
        ---------------
        :return ftbline: str
    """
    line = "ftm:{:.4f} shots:{}".format(frequency, shots)
    for key, value in kwargs.items():
        line += " {}:{}".format(key, value)
    line += "\n"
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
        "dipole": 1.0,
        "magnet": "false",
        "drpower": "10",
        "skiptune": "false",
    }

    param_dict.update(kwargs)
    for freq, shot in zip(frequencies, shotcounts):
        ftb_string += generate_ftb_str(freq, shot, **param_dict)
        if "magnet" in kwargs:
            param_dict["magnet"] = "true"
            ftb_string += generate_ftb_str(freq, shot, **param_dict)


def categorize_frequencies(
    frequencies,
    nshots=50,
    intensities=None,
    power=None,
    attn_list=None,
    dipole=None,
    attn=None,
    magnet=False,
    dr=False,
    discharge=False,
):
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

            ftb_str += generate_ftb_line(freq, shotcount, **{"skiptune": "false"})

            if dr is True:
                ftb_str += generate_ftb_line(
                    freq, shotcount, **{"skiptune": "true", "drfreq": dr_freq}
                )

            if dipole is True:
                for dipole_value in dipole_test:
                    ftb_str += generate_ftb_line(
                        freq, shotcount, **{dipole_flag: dipole_value}
                    )

            if magnet is True:
                ftb_str += generate_ftb_line(freq, shotcount, **{"magnet": "true"})

            if discharge is True:
                # Toggle the discharge stack on and off
                ftb_str += generate_ftb_line(
                    freq, shotcount, **{"pulse,1,enabled": "false"}
                )
                ftb_str += generate_ftb_line(
                    freq, shotcount, **{"pulse,1,enabled": "true"}
                )
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

    def dipole_analysis(
        self,
        batch_path,
        thres=0.5,
        dipoles=[1.0, 0.01, 0.1, 1.0, 3.0, 5.0],
        snr_thres=5.0,
    ):
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
            columns=["Frequency", "Dipole"],
        )
        full_df["Scan"] = scan_numbers

        # Loop over each scan, and determine whether or not there is a peak
        # If there is sufficiently strong feature, add it to the list with the
        # scan number
        detected_scans = list()
        for index, scan in enumerate(scan_numbers):
            scan_slice = batch_df.loc[
                (batch_df["Scan"] >= scan - 0.5) & (batch_df["Scan"] <= scan + 0.5)
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
            slice_df = obs_df.loc[obs_df["Frequency"] == frequency]
            # Sort the best response dipole moment at the top
            slice_df.sort_values(["SNR"], inplace=True, ascending=False)
            slice_df.index = np.arange(len(slice_df))
            optimal_data.append(slice_df.iloc[0].values)
        optimal_df = pd.DataFrame(
            optimal_data, columns=["Frequency", "Dipole", "Scan", "SNR"]
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
                transparent=True,
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
                dataframe["SNR"].values, nshots
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
                "cal_shots": dataframe["Shots"][0],
            }
            if "Dipole" in dataframe:
                cal_settings["cal_dipole"] = dataframe["Dipole"][0]
            cal_settings.update(kwargs)
            # Generate the calibration string
            cal_str = generate_ftb_line(
                cal_settings["cal_freq"],
                cal_settings["cal_shots"],
                **{"dipole": cal_settings["cal_dipole"], "skiptune": "false"},
            )
            cal_str = cal_str.replace("\n", " cal\n")

        # generate the batch file
        for index, row in dataframe.iterrows():
            # If we want to add a calibration line
            if index % cal_settings["cal_rate"] == 0:
                ftb_str += cal_str
            param_dict = {}
            # Make sure the magnet is off
            if "Dipole" in row:
                param_df["dipole"] = row["Dipole"]
            param_df["magnet"] = "false"
            param_df["skiptune"] = "false"
            ftb_str += generate_ftb_line(row["Frequency"], row["Shots"], **param_df)
            # Turn on the magnet
            param_df["skiptune"] = "true"
            param_df["magnet"] = "true"
            ftb_str += generate_ftb_line(row["Frequency"], row["Shots"], **param_df)

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
                ftb_str += generate_ftb_line(
                    freq,
                    nshots,
                    **{
                        "dipole": combo[0][1],
                        "pulse,{},enable".format(dr_channel): "false",
                        "skiptune": "false",
                    },
                )
            # Only do DR if the DR frequency is significantly different from
            # the cavity frequency
            if np.abs(combo[1][0] - freq) >= 100.0:
                ftb_str += generate_ftb_line(
                    freq,
                    nshots,
                    **{
                        "dipole": combo[0][1],
                        "drfreq": combo[1][0],
                        "pulse,{},enable".format(dr_channel): "true",
                        "skiptune": "true",
                    },
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
        progressions = analysis.harmonic_finder(self.dipole_df["Frequency"].values)
        self.progression_df = fitting.harmonic_fitter(progressions)
        data, ap_obj = analysis.cluster_AP_analysis(
            self.progression_df, True, False, **kwargs
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
                count += 1
                if (sub_index == 0) or (last_freq != pair[0][0]):
                    ftb_str += generate_ftb_line(
                        pair[0][0],
                        10,
                        **{
                            "dipole": pair[0][1],
                            "pulse,{},enable".format(dr_channel): "false",
                            "skiptune": "false",
                        },
                    )
                # Perform the DR measurement
                ftb_str += generate_ftb_line(
                    pair[0][0],
                    10,
                    **{
                        "dipole": pair[0][1],
                        "pulse,{},enable".format(dr_channel): "true",
                        "skiptune": "true",
                    },
                )
                last_freq = pair[0][0]
        print("There are {} combinations to test.".format(count))

        with open(
            "./ftbfiles/{}-progressionDR.ftb".format(self.exp_id), "w+"
        ) as write_file:
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

        fig.add_scatter(x=self.data["Scan"], y=self.data["Intensity"])

        if scan_number is not None:
            fig.add_bar(x=[scan_number], y=[np.max(self.data["Intensity"])])

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
            (dataframe["Scan"] >= scan_number - 1.5)
            & (dataframe["Scan"] <= scan_number + 1.5)
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
                transparent=True,
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
        routines.save_obj(self.__dict__, filepath)
        print("Saved session to {}".format(filepath))


def predict_prolate_series(progressions, J_thres=0.1):
    fit_df, fits = fitting.harmonic_fitter(progressions, J_thres)
    J_model = LinearModel()
    BJ_model = fitting.BJModel()
    predictions = dict()
    for index, row in fit_df.iterrows():
        row = row.dropna()
        J_values = row[[col for col in row.keys() if "J" in str(col)]].values
        if len(J_values) > 2:
            J_fit = J_model.fit(data=J_values, x=np.arange(len(J_values)))
            J_predicted = J_fit.eval(x=np.arange(-10, 10, 1))
            BJ_params = row[["B", "D"]].values
            freq_predicted = BJ_model.eval(
                J=J_predicted, B=BJ_params[0], D=BJ_params[1]
            )
        elif len(J_values) == 2:
            frequencies = row[[2, 4]].values
            approx_B = np.abs(np.diff(frequencies))
            next_freq = np.max(frequencies) + approx_B
            low_freq = np.min(frequencies) - approx_B
            freq_predicted = np.concatenate(
                (frequencies, [next_freq, low_freq]), axis=None
            )
            freq_predicted = np.sort(freq_predicted)
            J_predicted = freq_predicted / approx_B
        # Filter out negative frequencies
        freq_predicted = freq_predicted[0.0 < freq_predicted]
        predictions[index] = {
            "predicted_freq": freq_predicted,
            "predicted_J": J_predicted,
        }
    return predictions


@dataclass
class BlackchirpExperiment:
    exp_id: int
    fid_start: int = 0
    fid_end: int = -1
    ft_min: float = 0.0
    ft_max: float = 40000.0
    ft_filter: str = "boxcar"
    freq_offset: float = 0.0
    fids: List = field(default_factory=list)
    header: Dict = field(default_factory=dict)

    @classmethod
    def from_dir(cls, filepath):
        exp_id, header, fids, timedata = parsers.parse_blackchirp(filepath)
        exp_obj = cls(exp_id=exp_id, header=header, fids=fids)
        return exp_obj

    def process_ffts(self, weighting=None):
        """
        Batch perform FFTs on all of the FIDs. The end result is a Pandas DataFrame with the Frequency and Intensity
        data, where the intensity is just the weighted co-average of all the FFTs. By default, every FID is equally
        weighted. 
        Parameters
        ----------
        weighting

        Returns
        -------

        """
        weight_factors = {index: 1.0 for index in range(len(self.fids))}
        if weighting:
            weight_factors.update(**weighting)
        # Work out the frequency bins
        frequency = self.fids[0].determine_frequencies()
        # Weight the FIDs
        weighted_fids = [
            self.fids[index][1] * weight for index, weight in weight_factors.items()
        ]
        averaged = np.sum(weighted_fids) / np.sum(
            [weight for weight in weight_factors.values()]
        )
        # Calculate the sample rate; inverse of the spacing, converted back to seconds
        rate = 1.0 / self.header["spacing"] / 1e6
        fid2fft(averaged, rate, frequency)

        spectrum_df = pd.DataFrame(
            {"Frequency": fft_data[0][0] + self.freq_offset, "Intensity": averaged}
        )
        self.spectrum = spectrum_df
        return spectrum_df


@dataclass
class BlackChirpFid:
    xy_data: np.array
    header: Dict = field(default_factory=dict)

    @classmethod
    def from_binary(cls, filepath):
        """
        Create a BlackChirp FID object from a binary BlackChirp FID file.

        Parameters
        ----------
        filepath - str
            Filepath to the BlackChirp .fid file

        Returns
        -------
        BlackChirpFid object
        """
        param_dict, xy_data, _ = parsers.read_binary_fid(filepath)
        fid_obj = cls(xy_data, param_dict)
        return fid_obj

    def to_pickle(self, filepath, **kwargs):
        """
        Save the Blackchirp FID to a pickle file.

        Parameters
        ----------
        filepath - str
            Filepath to save the FID to
        kwargs - dict-like
            Additional keyword arguments that are passed to the
            pickle function.

        """
        routines.save_obj(self, filepath, **kwargs)

    def perform_fft(self, start=0, stop=-1, window="boxcar", f_min=0.0, f_max=30000.0):
        """
        Perform an FFT on the current FID to get the frequency domain spectrum.
        All of the arguments are optional, and provide control over how the FFT is performed, as well as post-processing
        parameters like window functions and zero-padding.

        This is based on the FFT code by Kyle Crabtree, with modifications to fit this dataclass.

        Parameters
        ----------
        start - int, optional
            Starting index for the FID array to perform the FFT
        stop - int, optional
            End index for the FID array to perform the FFT
        zpf - int, optional
            Pad the FID with zeros to nth nearest power of 2
        window - str
            Specify the window function used to process the FID. Defaults to boxcar, which is effectively no filtering.
            The names of the window functions available can be found at:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html
        f_min - float
            Specify the minimum frequency in the spectrum; everything below this value is set to zero
        f_max - float
            Specify the maximum frequency in the spectrum; everything above this value is set to zero

        Returns
        -------
        """
        fid = np.copy(self.xy_data[1])
        if window is not None and window in spsig.windows.__all__:
            window_f = spsig.windows.get_window(window, fid.size)
            fid *= window_f
        else:
            raise Exception("Specified window function is not implemented in SciPy!")
        # Set values to zero up to starting index
        fid[:start] = 0.0
        if stop < 0:
            # If we're using negative indexes
            fid[fid.size + stop :] = 0.0
        else:
            # Otherwise, index with a positive number
            fid[stop:] = 0.0
        # Perform the FFT
        fft = np.fft.rfft(fid)
        read_length = len(fid) // 2 + 1
        df = 1.0 / fid.size / self.header["spacing"]
        # Generate the frequency array
        frequency = np.linspace(0.0, self.header["sideband"] * df, read_length)
        frequency += self.header["probe_freq"]
        fft[(frequency >= f_max) & (frequency <= f_min)] = 0.0
        fft *= 1000.0
        return frequency, fft

    def determine_frequencies(self):
        """
        Calculate the frequency bins for the FFT.

        Returns
        -------
        frequency - numpy 1D array
            Array containing the frequency bins (x values)
        """
        fid = self.xy_data[1]
        df = 1.0 / fid.size / self.header["spacing"]
        read_length = len(fid) // 2 + 1
        # Generate the frequency array
        frequency = np.linspace(0.0, self.header["sideband"] * df, read_length)
        frequency += self.header["probe_freq"]
        return frequency

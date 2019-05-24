"""
    transition.py

    Contains the Transition, AssignmentSession, and Session classes that are designed to handle and assist
    the assignment of broadband spectra from the laboratory or astronomical observations.
"""

import os
from shutil import rmtree
from dataclasses import dataclass, field
from typing import List, Dict
from collections import OrderedDict
from copy import deepcopy
from itertools import product
import logging

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from lmfit.models import GaussianModel
from IPython.display import display, HTML
from periodictable import formula
from plotly.offline import plot
from plotly import graph_objs as go
from uncertainties import ufloat

from pyspectools import routines, parsers, figurefactory
from pyspectools import ftmw_analysis as fa
from pyspectools import fitting
from pyspectools import units
from pyspectools import database
from pyspectools.astro import analysis as aa
from pyspectools.spectra import analysis


@dataclass
class Transition:
    """
        DataClass for handling assignments.
        Attributes are assigned in order to be sufficiently informative for a
        line assignment to be unambiguous and reproduce it later in a form
        that is both machine and human readable.

        Attributes
        ----------
        name : str
            IUPAC/common name; the former is preferred to be unambiguous
        formula : str
            Chemical formula, or usually the stochiometry
        smiles : str
            SMILES code that provides a machine and human readable chemical specification
        frequency : float
            Observed frequency in MHz
        intensity : float
            Observed intensity, in whatever units the experiments are in. Examples are Jy/beam, or micro volts.
        catalog_frequency : float
            Catalog frequency in MHz
        catalog_intensity : float
            Catalog line intensity, typically in SPCAT units
        S : float
            Theoretical line strength; differs from the catalog line strength as it may be used for intrinsic line
            strength S u^2
        peak_id : int
            Peak id from specific experiment
        uline : bool
            Flag to indicate whether line is identified or not
        composition : list of str
            A list of atomic symbols specifying what the experimental elemental composition is. Influences which
            molecules are considered possible in the Splatalogue assignment procedure.
        v_qnos : list of int
            Quantum numbers for vibrational modes. Index corresponds to mode, and int value to number of quanta.
            Length should be equal to 3N-6.
        r_qnos : str
            Rotational quantum numbers. TODO - better way of managing rotational quantum numbers
        experiment : int
            Experiment ID to use as a prefix/suffix for record keeping
        weighting : float
            Value for weighting factor used in the automated assignment
        fit : dict
            Contains the fitted parameters and model
        ustate_energy : float
            Energy of the upper state in Kelvin
        lstate_energy: float
            Energy of the lower state in Kelvin
        intereference : bool
            Flag to indicate if this assignment is not molecular in nature
        source : str
            Indicates what the source used for this assignment is
        public : bool
            Flag to indicate if the information for this assignment is public/published
        velocity : float
            Velocity of the source used to make the assignment in km/s
    """
    name: str = ""
    smiles: str = ""
    formula: str = ""
    frequency: float = 0.0
    catalog_frequency: float = 0.0
    catalog_intensity: float = 0.0
    deviation: float = 0.0
    intensity: float = 0.0
    S: float = 0.0
    peak_id: int = 0
    experiment: int = 0
    uline: bool = True
    composition: List[str] = field(default_factory=list)
    v_qnos: List[int] = field(default_factory=list)
    r_qnos: str = ""
    fit: Dict = field(default_factory=dict)
    ustate_energy: float = 0.0
    lstate_energy: float = 0.0
    interference: bool = False
    weighting: float = 0.0
    source: str = "Catalog"
    public: bool = True
    velocity: float = 0.

    def __eq__(self, other):
        """ Dunder method for comparing molecules.
            This method is simply a shortcut to see if
            two molecules are the same based on their
            SMILES code, the chemical name, and frequency.
        """
        if type(self) == type(other):
            comparisons = [
                self.smiles == other.smiles,
                self.name == other.name,
                self.frequency == other.frequency,
                self.v_qnos == other.v_qnos
            ]
            return all(comparisons)
        else:
            return False

    def __str__(self):
        """
        Dunder method for representing an Transition, which returns
        the name of the line and the frequency.

        Returns
        -------
        str
            name and frequency of the Transition
        """
        return f"{self.name}, {self.frequency}"

    def calc_intensity(self, Q, T=300.):
        """
        Convert linestrength into intensity.

        Parameters
        ----------
        Q - float
            Partition function for the molecule at temperature T
        T - float
            Temperature to calculate the intensity at in Kelvin

        Returns
        -------
        I - float
            log10 of the intensity in SPCAT format
        """
        # Take the frequency value to calculate I
        frequency = max([self.frequency, self.catalog_frequency])
        I = units.S2I(
            self.intensity,
            Q,
            frequency,
            units.calc_E_lower(frequency, self.ustate_energy),
            T
        )
        self.S = I
        return I


    def calc_linestrength(self, Q, T=300.):
        """
        Convert intensity into linestrength.

        Parameters
        ----------
        Q - float
            Partition function for the molecule at temperature T
        T - float
            Temperature to calculate the intensity at in Kelvin

        Returns
        -------
        intensity - float
            intrinsic linestrength of the transition
        """
        # Take the frequency value to calculate I
        frequency = max([self.frequency, self.catalog_frequency])
        intensity = units.I2S(
            self.S,
            Q,
            frequency,
            units.calc_E_lower(frequency, self.ustate_energy),
            T
        )
        self.intensity = intensity
        return intensity


    def to_file(self, filepath, format="yaml"):
        """
        Save an Transition object to disk with a specified file format.
        Defaults to YAML.

        Parameters
        ----------
        filepath : str
            Path to yaml file
        format : str, optional
            Denoting the syntax used for dumping. Defaults to YAML.
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

    def get_spectrum(self, x):
        """
        Generate a synthetic peak by supplying
        the x axis for a particular spectrum. This method
        assumes that some fit parameters have been determined
        previously.

        Parameters
        ----------
        x : Numpy 1D array
            Frequency bins from an experiment to simulate the line features.

        Returns
        -------
        Numpy 1D array
            Values of the model function spectrum at each particular value of x
        """
        if hasattr(self, "fit"):
            return self.fit.eval(x=x)
        else:
            raise Exception("get_spectrum() with no fit data available!")

    @classmethod
    def from_dict(obj, data_dict):
        """ 
        Method for generating an Assignment object
        from a dictionary. All this method does is
        unpack a dictionary into the __init__ method.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing all of the Assignment DataClass fields that are to be populated.

        Returns
        -------
        Transition
            Converted Assignment object from the input dictionary
        """
        assignment_obj = obj(**data_dict)
        return assignment_obj

    @classmethod
    def from_yml(obj, yaml_path):
        """
        Method for initializing an Assignment object from a YAML file.

        Parameters
        ----------
        yaml_path : str
            path to yaml file

        Returns
        -------
        Transition
            Assignment object loaded from a YAML file.
        """
        yaml_dict = routines.read_yaml(yaml_path)
        assignment_obj = obj(**yaml_dict)
        return assignment_obj

    @classmethod
    def from_json(obj, json_path):
        """
        Method for initializing an Assignment object from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to JSON file

        Returns
        -------
        Transition
            Assignment object loaded from a JSON file.
        """
        json_dict = routines.read_json(json_path)
        assignment_obj = obj(**json_dict)
        return assignment_obj


@dataclass
class Session:
    """ 
    DataClass for a Session, which simply holds the
    experiment ID, composition, and guess_temperature.
    Doppler broadening can also be incorporated. 

    Attributes
    ----------
    experiment : int
        ID for experiment
    composition : list of str
        List of atomic symbols. Used for filtering out species in the Splatalogue assignment procedure.
    temperature : float
        Temperature in K. Used for filtering transitions in the automated assigment, which are 3 times this value.
    doppler : float
        Doppler width in km/s; default value is about 5 kHz at 15 GHz. Used for simulating lineshapes and
        for lineshape analysis.
    velocity : float
        Radial velocity of the source in km/s; used to offset the frequency spectrum
    freq_prox : float
        frequency cutoff for line assignments. If freq_abs attribute is True, this value is taken as the absolute value.
        Otherwise, it is a percentage of the frequency being compared.
    freq_abs : bool
        If True, freq_prox attribute is taken as the absolute value of frequency, otherwise as a decimal percentage of
        the frequency being compared.
    baseline : float
        Baseline level of signal used for intensity calculations and peak detection
    noise_rms : float
        RMS of the noise used for intensity calculations and peak detection
    noise_region : 2-tuple of floats
        The frequency region used to define the noise floor.
    """
    experiment: int
    composition: List[str] = field(default_factory=list)
    temperature: float = 4.0
    doppler: float = 0.01
    velocity: float = 0.
    freq_prox: float = 0.1
    freq_abs: bool = True
    baseline: float = 0.
    noise_rms: float = 0.
    noise_region: List[float] = field(default_factory=list)

    def __str__(self):
        form = "Experiment: {}, Composition: {}, Temperature: {} K".format(
            self.experiment, self.composition, self.temperature
        )
        return form


class AssignmentSession:
    """ Class for managing a session of assigning molecules
        to a broadband spectrum.

        Wraps some high level functionality from the analysis
        module so that this can be run reproducibly in a jupyter
        notebook.

        TODO - Homogenize the assignment functions to use one main
               function, as opposed to having separate functions
               for catalogs, lin, etc.
    """

    @classmethod
    def load_session(cls, filepath):
        """
            Load an AssignmentSession from disk, once it has
            been saved with the save_session method which creates a pickle
            file.

            Parameters
            --------------
            filepath : str
                path to the AssignmentSession pickle file; typically in the sessions/{experiment_id}.pkl

            Returns
            --------------
            AssignmentSession
                Instance of the AssignmentSession loaded from disk
        """
        session = routines.read_obj(filepath)
        session._init_logging()
        session.logger.info("Reloading session: {}".format(filepath))
        return session

    @classmethod
    def from_ascii(
            cls, filepath, experiment, composition=["C", "H"], delimiter="\t", temperature=4.0, velocity=0.,
            col_names=None, freq_col="Frequency", int_col="Intensity", skiprows=0, verbose=True, **kwargs
    ):
        """

        Parameters
        ----------
        filepath : str
            Filepath to the ASCII spectrum
        experiment : int
            Integer identifier for the experiment
        composition : list of str
            List of atomic symbols, representing the atomic composition of the experiment
        delimiter : str
            Delimiter character used in the ASCII file. For example, "\t", "\s", ","
        temperature : float
            Rotational temperature in Kelvin used for the experiment
        header : list of str, optional
            Names of the columns
        freq_col : str
            Name of the column to be used for the frequency axis
        int_col : str
            Name of the column to be used for the intensity axis
        kwargs
            Additional kwargs are passed onto initializing the Session class

        Returns
        -------
        AssignmentSession
        """
        spec_df = parsers.parse_ascii(filepath, delimiter, col_names, skiprows=skiprows)
        session = cls(spec_df, experiment, composition, temperature, velocity, freq_col, int_col, verbose, **kwargs)
        return session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, handler in self.log_handlers.items():
            handler.close()

    def __init__(
            self, exp_dataframe, experiment, composition, temperature=4.0, velocity=0.,
            freq_col="Frequency", int_col="Intensity", verbose=True, **kwargs
    ):
        """ init method for AssignmentSession.

            Required arguments are necessary metadata for controlling various aspects of
            the automated assignment procedure, as well as for reproducibility.

            Parameters
            -------------------------
             exp_dataframe : pandas dataframe
                Dataframe with observational data in frequency/intensity
             experiment : int
                ID for the experiment
             composition : list of str
                Corresponds to elemental composition composition; e.g. ["C", "H"]. Used for splatalogue analysis.
             freq_col : str, optional
                Specifying the name for the frequency column
             int_col: optional str arg specifying the name of the intensity column
        """
        # Make folders for organizing output
        folders = ["assignment_objs", "queries", "sessions", "clean", "figures", "reports", "logs", "outputs", "ftb"]
        for folder in folders:
            if os.path.isdir(folder) is False:
                os.mkdir(folder)
        # Initialize a Session dataclass
        self.session = Session(experiment, composition, temperature, velocity=velocity)
        # Update additional setttings
        self.session.__dict__.update(**kwargs)
        self.data = exp_dataframe
        # Set the temperature threshold for transitions to be 3x the set value
        self.t_threshold = self.session.temperature * 3.
        self.assignments = list()
        self.ulines = OrderedDict()
        self.umols = list()
        self.verbose = verbose
        self._init_logging()
        #self.umol_counter = self.umol_gen()
        # Default settings for columns
        if freq_col not in self.data.columns:
            self.freq_col = self.data.columns[0]
        else:
            self.freq_col = freq_col
        if int_col not in self.data.columns:
            self.int_col = self.data.columns[1]
        else:
            self.int_col = int_col
        if velocity != 0.:
            self.set_velocity(velocity)

    def __truediv__(self, other, copy=True):
        """
        Method to divide the spectral intensity of the current experiment by another.

        If the copy keyword is True, this method creates a deep copy of the current experiment and returns the copy
        with the updated intensities. Otherwise, the spectrum of the current experiment is modified.

        Parameters
        ----------
        other - AssignmentSession object
            Other AssignmentSession object to compare spectra with. Acts as denominator.
        copy - bool, optional
            If True, returns a copy of the experiment, with the ID number added

        Returns
        -------
        new_experiment - AssignmentSession object
            Generated new experiment with the divided spectrum, if copy is True
        """
        if copy is True:
            new_experiment = deepcopy(self)
            new_experiment.data[:, self.int_col] = new_experiment[self.int_col] / other.data[other.int_col]
            new_experiment.session.id += other.session.id
            return new_experiment
        else:
            self.data[:, self.int_col] = self.data[self.int_col] / other.data[other.int_col]

    def umol_gen(self):
        """
        Method for keeping track of what unidentified molecule we're up to. Currently not used.

        Yields
        ------
        str
            Formatted as "UMol_XXX"
        """
        counter = 1
        while counter <= 200:
            yield "UMol_{:03.d}".format(counter)
            counter+=1

    def _init_logging(self):
        """
        Set up the logging formatting and files. The levels are defined in the dictionary mapping, and so new
        levels and their warning level should be defined there. Additional log files can be added in the
        log_handlers dictionary. In the other routines, the logger attribute should be used.
        """
        mapping = {
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
            "analysis": logging.INFO,
            "stream": logging.INFO
        }
        logging.captureWarnings(True)
        self.logger = logging.getLogger("{} log".format(self.session.experiment))
        self.logger.setLevel(logging.DEBUG)
        # Define file handlers for each type of log
        self.log_handlers = {
            "analysis": logging.FileHandler("./logs/{}-analysis.log".format(self.session.experiment)),
            "warning": logging.FileHandler("./logs/{}-warnings.log".format(self.session.experiment)),
            "debug": logging.FileHandler("./logs/{}-debug.log".format(self.session.experiment))
        }
        if self.verbose is True:
            self.log_handlers["stream"] = logging.StreamHandler()
        # Set up the formatting and the level definitions
        for key, handler in self.log_handlers.items():
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            # do not redefine the default warning levels
            handler.setLevel(mapping[key])
            self.logger.addHandler(handler)

    def set_velocity(self, value):
        """
        Set the radial velocity offset for the spectrum. The velocity is specified in km/s, and is set up
        such that the notation is positive velocity yields a redshifted spectrum (i.e. moving towards us).

        This method should be used to change the velocity, as it will automatically re-calculate the dataframe
        frequency column to the new velocity.

        Parameters
        ----------
        value : float
            Velocity in km/s
        """

        # Assume that if this is the first time we're performing a frequency offset, make a copy of the data
        if "Laboratory Frame" not in self.data.columns:
            self.data.loc[:, "Laboratory Frame"] = self.data.loc[:, self.freq_col]
        else:
            # If we have performed the shift before, copy back the unshifted data
            self.data.loc[:, self.freq_col] = self.data.loc[:, "Laboratory Frame"]
        doppler_offset = units.dop2freq(value, self.data["Laboratory Frame"].values)
        # Offset the values
        self.data.loc[:, self.freq_col] += doppler_offset
        self.session.velocity = value
        self.logger.info("Set the session velocity to {}.".format(value))

    def detect_noise_floor(self, region=None):
        """
        Set the noise parameters for the current spectrum. Control over what "defines" the noise floor
        is specified with the parameter region. By default, if region is None then the function will
        perform an initial peak find using 1% of the maximum intensity as the threshold. The noise region
        will be established based on the largest gap between peaks, i.e. hopefully capturing as little
        features in the statistics as possible.

        Parameters
        ----------
        region - 2-tuple or None, optional
            If None, use the automatic algorithm. Otherwise, a 2-tuple specifies the region of the spectrum
            in frequency to use for noise statistics.

        Returns
        -------
        baseline - float
            Value of the noise floor
        rms - float
            Noise RMS/standard deviation
        """
        if region is None:
            # Perform rudimentary peak detection
            threshold = 0.2 * self.data[self.int_col].max()
            peaks_df = analysis.peak_find(
                self.data,
                freq_col=self.freq_col,
                int_col=self.int_col,
                thres=threshold,
            )
            if len(peaks_df) >= 2:
                # find largest gap in data, including the edges of the spectrum
                freq_values = list(peaks_df["Frequency"].values)
                freq_values.extend(
                    [self.data[self.freq_col].min(), self.data[self.freq_col].max()]
                    )
                # sort the values
                freq_values = sorted(freq_values)
                self.logger.info("Possible noise regio:")
                self.logger.info(freq_values)
                index = np.argmax(np.diff(freq_values))
                # Define region as the largest gap
                region = freq_values[index:index+2]
                # Add a 30 MHz offset to either end of the spectrum
                region[0] = region[0] + 30.
                region[1] = region[1] - 30.
                # Make sure the frequencies are in ascending order
                region = np.sort(region)
                self.logger.info("Noise region defined as {} to {}.".format(*region))
                noise_df = self.data.loc[
                    self.data[self.freq_col].between(*region)
                ]
                if len(noise_df) < 50:
                    noise_df = self.data.sample(int(len(self.data) * 0.1))
                    self.logger.warning("Noise region too small; taking a statistical sample.")
            else:
                # If we haven't found any peaks, sample 10% of random channels and determine the
                # baseline from those values
                noise_df = self.data.sample(int(len(self.data) * 0.1))
                self.logger.warning("No obvious peaks detected; taking a statistical sample.")
        # Calculate statistics
        baseline = np.average(noise_df[self.int_col])
        rms = np.std(noise_df[self.int_col])
        self.session.noise_rms = rms
        self.session.baseline = baseline
        self.session.noise_region = region
        self.logger.info("Baseline signal set to {}.".format(baseline))
        self.logger.info("Noise RMS set to {}.".format(rms))
        return baseline, rms

    def find_peaks(self, threshold=None, region=None, sigma=6):
        """
            Find peaks in the experiment spectrum, with a specified threshold value or automatic threshold.
            The method calls the peak_find function from the analysis module, which in itself wraps peakutils.

            The function works by finding regions of the intensity where the first derivative goes to zero
            and changes sign. This gives peak frequency/intensities from the digitized spectrum, which is
            then "refined" by interpolating over each peak and fitting a Gaussian to determine the peak.

            The peaks are then returned as a pandas DataFrame, which can also be accessed in the peaks_df
            attribute of AssignmentSession.

            Parameters
            ----------
             threshold: float or None
                Peak detection threshold. If None, will take 1.5 times the noise RMS.
             region - 2-tuple or None, optional
                If None, use the automatic algorithm. Otherwise, a 2-tuple specifies the region of the spectrum
                in frequency to use for noise statistics.
             sigma - float, optional
                Defines the number of sigma (noise RMS) above the baseline to use as the peak detection threshold.

            Returns
            -------
            peaks_df : dataframe
                Pandas dataframe with Frequency/Intensity columns, corresponding to peaks
        """
        if threshold is None:
            # Use a quasi-intelligent method of determining the noise floor
            # and ultimately using noise + 1 sigma
            baseline, rms = self.detect_noise_floor(region)
            threshold = baseline + (rms * sigma)
        self.threshold = threshold
        self.logger.info("Peak detection threshold is: {}".format(threshold))
        peaks_df = analysis.peak_find(
            self.data,
            freq_col=self.freq_col,
            int_col=self.int_col,
            thres=threshold,
        )
        # Shift the peak intensities down by the noise baseline
        peaks_df.loc[:, self.int_col] = peaks_df[self.int_col] - self.session.baseline
        self.logger.info("Found {} peaks in total.".format(len(peaks_df)))
        # Reindex the peaks
        peaks_df.reset_index(drop=True, inplace=True)
        if len(peaks_df) != 0:
            # Generate U-lines
            self.df2ulines(peaks_df, self.freq_col, self.int_col)
            # Assign attribute
            self.peaks = peaks_df
            self.peaks.to_csv("./outputs/{}-peaks.csv".format(self.session.experiment), index=False)
            return peaks_df
        else:
            return None

    def df2ulines(self, dataframe, freq_col=None, int_col=None):
        """
        Add a dataframe of frequency and intensities to the session U-line dictionary. This function provides more
        manual control over what can be processed in the assignment pipeline, as not everything can be picked
        up by the peak finding algorithm.

        Parameters
        ----------
        dataframe : pandas dataframe
            Dataframe containing a frequency and intensity column to add to the uline list.
        freq_col : None or str
            Specify column to use for frequencies. If None, uses the session value freq_col.
        int_col : None or str
            Specify column to use for intensities. If None, uses the session value int_col.
        """
        if freq_col is None:
            freq_col = self.freq_col
        if int_col is None:
            int_col = self.int_col
        self.logger.info("Adding additional U-lines based on user dataframe.")
        # This number is used to work out the index to carry on from
        total_num = len(self.assignments) + len(self.ulines)
        self.logger.info("So far, there are {} line entries in this session.".format(total_num))
        # Set up session information to be passed in the U-line
        skip = ["temperature", "doppler", "freq_abs", "freq_prox", "noise_rms", "baseline", "header", "noise_region"]
        selected_session = {
            key: self.session.__dict__[key] for key in self.session.__dict__ if key not in skip
            }
        for index, row in dataframe.iterrows():
            self.logger.info("Added U-line {}, frequency {:,.4f}".format(index + total_num + 1, row[freq_col]))
            ass_obj = Transition(
                frequency=row[freq_col],
                intensity=row[int_col],
                peak_id=index + total_num + 1,
                **selected_session
            )
            if ass_obj not in self.ulines.values() and ass_obj not in self.assignments:
                self.ulines[index + total_num] = ass_obj
        self.logger.info("There are now {} line entries in this session.".format(total_num + index))

    def search_frequency(self, frequency):
        """
        Function for searching the experiment for a particular frequency. The search range is defined by
        the Session attribute freq_prox, and will first look for the frequency in the assigned features
        if any have been made. The routine will then look for it in the U-lines.

        Parameters
        ----------
        frequency : float
            Center frequency in MHz

        Returns
        -------
        dataframe
            Pandas dataframe with the matches
        """
        slice_df = []
        if self.session.freq_abs is False:
            lower_freq = frequency * (1. - self.session.freq_prox)
            upper_freq = frequency * (1 + self.session.freq_prox)
        else:
            lower_freq = frequency - self.session.freq_prox
            upper_freq = frequency + self.session.freq_prox
        if hasattr(self, "table"):
            slice_df = self.table.loc[
                (self.table["frequency"] >= lower_freq) &
                (self.table["frequency"] <= upper_freq)
                ]
        # If no hits turn up, look for it in U-lines
        if len(slice_df) == 0:
            self.logger.info("No assignment found; searching U-lines")
            ulines = np.array([[index, uline.frequency] for index, uline in self.ulines.items()])
            nearest, array_index = routines.find_nearest(ulines[:,1], frequency)
            uline_index = int(ulines[array_index, 0])
            return nearest, uline_index
        else:
            self.logger.info("Found assignments.")
            return slice_df

    def in_experiment(self, frequency):
        """
        Method to ask a simple yes/no if the frequency exists in either U-lines or assignments.

        Parameters
        ----------
        frequency : float
            Center frequency to search for in MHz

        Returns
        -------
        True
            If the frequency is present in the experiment
        False
            If the frequency does not exist
        """
        try:
            slice_df = self.search_frequency(frequency)
            if len(slice_df) > 0:
                return True
            else:
                return False
        except:
            return False

    def splat_assign_spectrum(self, auto=False):
        """
        Alias for `process_splatalogue`. Function will be removed in a later version.

        Parameters
        ----------
        auto : bool
            Specifies whether the assignment procedure is automatic.
        """
        self.process_splatalogue(auto=auto)

    def process_splatalogue(self, auto=False):
        """ Function that will provide an "interface" for interactive
            line assignment in a notebook environment.

            Basic functionality is looping over a series of peaks,
            which will query splatalogue for known transitions in the
            vicinity. If the line is known in Splatalogue, it will
            throw it into an Transition object and flag it as known.
            Conversely, if it's not known in Splatalogue it will defer
            assignment, flagging it as unassigned and dumping it into
            the `uline` attribute.

        Parameters
        ----------
         auto: bool
             If True the assignment process does not require user input, otherwise will prompt user.
        """
        if hasattr(self, "peaks") is False:
            self.logger.warning("Peak detection not run; running with default settings.")
            self.find_peaks()
        self.logger.info("Beginning Splatalogue lookup on {} lines.".format(len(self.peaks)))
        for uindex, uline in tqdm(list(self.ulines.items())):
            frequency = uline.frequency
            self.logger.info("Searching for frequency {:,.4f}".format(frequency))
            # Call splatalogue API to search for frequency
            if self.session.freq_abs is True:
                width = self.session.freq_prox
            else:
                width = self.session.freq_prox * frequency
            splat_df = analysis.search_center_frequency(frequency, width=width)
            if splat_df is not None:
                # Filter out lines that are way too unlikely on grounds of temperature
                splat_df = splat_df.loc[splat_df["E_U (K)"] <= self.t_threshold]
                # Filter out quack elemental compositions
                for index, row in splat_df.iterrows():
                    # Convert the string into a chemical formula object
                    try:
                        # Clean vibrational state and torsional specification
                        cation_flag = False
                        clean_formula = row["Species"].split("v")[0]
                        # Remove labels
                        for label in ["l-", "c-", "t-", ",", "-gauche", "cis-", "trans-", "trans", "anti", "sym", "="]:
                            clean_formula = clean_formula.replace(label, "")
                        # Remove the torsional states
                        clean_formula = clean_formula.split("-")[-1]
                        # These are typically charged, and the chemical fomrula parsing does not play well
                        if "+" in clean_formula:
                            cation_flag = True
                            clean_formula = clean_formula.replace("+", "")
                        formula_obj = formula(clean_formula)
                        # Add the charge back on
                        if cation_flag is True:
                            clean_formula += "+"
                        # Check if proposed molecule contains atoms not
                        # expected in composition
                        comp_check = all(
                            str(atom) in self.session.composition for atom in formula_obj.atoms
                        )
                        if comp_check is False:
                            # If there are crazy things in the mix, forget about it
                            self.logger.info("Molecule " + clean_formula + " rejected.")
                            splat_df.drop(index, inplace=True)
                    except:
                        self.logger.warning("Could not parse molecule " + clean_formula + " rejected.")
                        splat_df.drop(index, inplace=True)
                nitems = len(splat_df)

                splat_df = self.calc_line_weighting(
                    frequency, splat_df, prox=self.session.freq_prox, abs=self.session.freq_abs
                )
                if splat_df is not None:
                    self.logger.info("Found {} candidates for frequency {:,.4f}, index {}.".format(
                        len(splat_df), frequency, uindex)
                    )
                    if self.verbose is True:
                        display(HTML(splat_df.to_html()))
                    try:
                        if auto is False:
                            # If not automated, we need a human to look at frequencies
                            # Print the dataframe for notebook viewing
                            splat_index = int(
                                input(
                                    "Please choose an assignment index: 0 - " + str(nitems - 1)
                                )
                            )
                        else:
                            # If automated, choose closest frequency
                            splat_index = 0
                        self.logger.info("Index {} was chosen.".format(splat_index))
                        ass_df = splat_df.iloc[[splat_index]]
                        splat_df.to_csv(
                            "queries/{0}-{1}.csv".format(self.session.experiment, uindex), index=False
                        )
                        ass_dict = {
                            "uline": False,
                            "index": uindex,
                            "frequency": frequency,
                            "name": ass_df["Chemical Name"][0],
                            "catalog_frequency": ass_df["Frequency"][0],
                            "catalog_intensity": ass_df["CDMS/JPL Intensity"][0],
                            "formula": ass_df["Species"][0],
                            "r_qnos": ass_df["Resolved QNs"][0],
                            #"S": ass_df["$S_{ij}^2 D^2$"][0],
                            "ustate_energy": ass_df["E_U (K)"][0],
                            "weighting": ass_df["Weighting"][0],
                            "source": "CDMS/JPL",
                            "deviation": frequency - ass_df["Frequency"][0]
                        }
                        # Need support to convert common name to SMILES
                        self._assign_line(**ass_dict)
                    except ValueError:
                        # If nothing matches, keep in the U-line
                        # pile.
                        self.logger.info("Deferring assignment for index {}.".format(uindex))
                else:
                    # Throw into U-line pile if no matches at all
                    self.logger.info("No species known for {:,.4f}".format(frequency))
        self.logger.info("Splatalogue search finished.")

    def process_lin(self, name, formula, linpath, auto=True, **kwargs):
        """
            Reads in a line file and sweeps through the U-line list.

            Operationally, the same as the catalog and splatalogue methods,
            but parses a line file instead. The differences are a lack of
            filtering, since there is insufficient information in a lin
            file.

            Kwargs are passed to the `assign_line` function, which provides
            the user with additional options for flagging an Transition
            (e.g. public, etc.)

            Parameters
            ----------
             name: str
                Common name of the molecule
             formula: str
                Chemical formula of the molecule
             linpath: str
                Path to line file to be parsed
             auto: bool, optional
                Specify whether assignment is non-interactive.
        """
        old_nulines = len(self.ulines)
        lin_df = parsers.parse_lin(linpath)

        self.logger.info("Processing .lin file for {}, filepath: {}".format(name, linpath))
        for uindex, uline in tqdm(list(self.ulines.items())):
            sliced_catalog = self.calc_line_weighting(
                uline.frequency,
                lin_df,
                prox=self.session.freq_prox,
                abs=self.session.freq_abs
            )
            if sliced_catalog is not None:
                if self.verbose is True:
                    display(HTML(sliced_catalog.to_html()))
                if auto is False:
                    index = int(input("Please choose a candidate by index"))
                elif auto is True:
                    index = 0
                if index in sliced_catalog.index:
                    select_df = sliced_catalog.iloc[index]
                    qnos = "".join(sliced_catalog["Quantum numbers"])
                    assign_dict = {
                        "name": name,
                        "formula": formula,
                        "index": uindex,
                        "frequency": uline.frequency,
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "source": "Line file",
                        "deviation": uline.frequency - select_df["Frequency"]
                    }
                    assign_dict.update(**kwargs)
                    self._assign_line(**assign_dict)
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        self.logger.info("Prior number of ulines: {}".format(old_nulines))
        self.logger.info("Current number of ulines: {}".format(len(self.ulines)))
        self.logger.info("Finished looking up lin file.")

    def calc_line_weighting(
            self, frequency, catalog_df, prox=0.00005,
            abs=True, freq_col="Frequency", int_col="Intensity"
    ):
        """
            Function for calculating the weighting factor for determining
            the likely hood of an assignment. The weighting factor is
            determined by the proximity of the catalog frequency to the
            observed frequency, as well as the theoretical intensity if it
            is available.

            Parameters
            ----------------
             frequency : float
                Observed frequency in MHz
             catalog_df : dataframe
                Pandas dataframe containing the catalog data entries
             prox: float, optional
                Frequency proximity threshold
             abs: bool
                Specifies whether argument prox is taken as the absolute value

            Returns
            ---------------
            None
                If nothing matches the frequency, returns None.
            dataframe
                If matches are found, calculate the weights and return the candidates in a dataframe.
        """
        if abs is False:
            lower_freq = frequency * (1 - prox)
            upper_freq = frequency * (1 + prox)
        else:
            lower_freq = frequency - prox
            upper_freq = frequency + prox
        sliced_catalog = catalog_df.loc[
                catalog_df[freq_col].between(lower_freq, upper_freq)
            ]
        nentries = len(sliced_catalog)
        if nentries > 0:
            if int_col in sliced_catalog:
                column = sliced_catalog[int_col]
            elif "CDMS/JPL Intensity" in sliced_catalog:
                column = sliced_catalog["CDMS/JPL Intensity"]
            else:
                column = None
            # Vectorized function for calculating the line weighting
            sliced_catalog["Weighting"] = analysis.line_weighting(
                frequency, sliced_catalog[freq_col], column
            )
            # Normalize the weights
            if nentries > 1:
                sliced_catalog.loc[:, "Weighting"] /= sliced_catalog["Weighting"].max()
                # Sort by obs-calc
                sliced_catalog.sort_values(["Weighting"], ascending=False, inplace=True)
            sliced_catalog.reset_index(drop=True, inplace=True)
            return sliced_catalog
        else:
            return None

    def process_catalog(self, name, formula, catalogpath, auto=True, thres=-10., progressbar=True, **kwargs):
        """
            Reads in a catalog (SPCAT) file and sweeps through the
            U-line list for this experiment finding coincidences.

            Similar to the splatalogue interface, lines are rejected
            on state energy from the t_threshold attribute.

            Each catalog entry will be weighted according to their
            theoretical intensity and deviation from observation. In
            automatic mode, the highest weighted transition is assigned.

            Kwargs are passed to the `assign_line` function, which will
            allow the user to provide additional flags/information for
            the Transition object.

            Parameters
            ----------------
             name : str
                Corresponds to common name of molecule
             formula : str
                Chemical formula or stochiometry
             catalogpath : str
                Filepath to the SPCAT file
             auto : bool, optional
                If True, assignment does not require user input
             thres : float
                log10 of the theoretical intensity to use as a bottom limit
        """
        old_nulines = len(self.ulines)
        self.logger.info("Processing catalog for {}, catalog path: {}".format(name, catalogpath))
        catalog_df = parsers.parse_cat(
            catalogpath,
            self.data[self.freq_col].min(),
            self.data[self.freq_col].max()
        )
        # Filter out the states with high energy
        catalog_df = catalog_df.loc[
            (catalog_df["Lower state energy"] <= self.t_threshold) &
            (catalog_df["Intensity"] >= thres)
            ]
        # Loop over the uline list
        iterate_list = list(self.ulines.items())
        if progressbar is True:
            iterate_list = tqdm(iterate_list)
        for uindex, uline in iterate_list:
            # 0.1% of frequency
            sliced_catalog = self.calc_line_weighting(
                uline.frequency, catalog_df, prox=self.session.freq_prox, abs=self.session.freq_abs
            )
            if sliced_catalog is not None:
                if self.verbose is True:
                    display(HTML(sliced_catalog.to_html()))
                if auto is False:
                    index = int(input("Please choose a candidate by index."))
                elif auto is True:
                    index = 0
                if index in sliced_catalog.index:
                    select_df = sliced_catalog.iloc[index]
                    # Create an approximate quantum number string
                    qnos = "N'={}, J'={}".format(*sliced_catalog[["N'", "J'"]].values[0])
                    qnos += "N''={}, J''={}".format(*sliced_catalog[["N''", "J''"]].values[0])
                    assign_dict = {
                        "name": name,
                        "formula": formula,
                        "index": uindex,
                        "frequency": uline.frequency,
                        "lstate_energy": select_df["Lower state energy"],
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "catalog_intensity": select_df["Intensity"],
                        "source": "Catalog",
                        "deviation": uline.frequency - select_df["Frequency"]
                    }
                    # Pass whatever extra stuff
                    assign_dict.update(**kwargs)
                    # Use assign_line function to mark peak as assigned
                    self._assign_line(**assign_dict)
                else:
                    raise IndexError("Invalid index chosen for assignment.")
        # Update the internal table
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        self.logger.info("Prior number of ulines: {}".format(old_nulines))
        self.logger.info("Current number of ulines: {}".format(len(self.ulines)))
        self.logger.info("Finished looking up catalog.")

    def process_frequencies(self, frequencies, ids, molecule=None, **kwargs):
        """
        Function to mark frequencies to belong to a single molecule, and for book-keeping's
        sake a list of ids are also required to indicate the original scan as the source
        of the information.

        Parameters
        ----------
         frequencies: list of frequencies associated with a molecule
         ids: list of scan IDs
         molecule: optional str specifying the name of the molecule
        """
        counter = 0
        self.logger.info("Processing {} frequencies for molecule {}.".format(len(frequencies), molecule))
        for freq, scan_id in zip(frequencies, ids):
            uline_freqs = np.array([uline.frequency for uline in self.ulines.values()])
            nearest, index = routines.find_nearest(uline_freqs, freq)
            # Find the nearest U-line, and if it's sufficiently close then
            # we assign it to this molecule. This is just to make sure that
            # if we sneak in some random out-of-band frequency this we won't
            # just assign it
            if self.session.freq_abs is True:
                thres = self.session.freq_prox
            else:
                thres = 0.1
            if np.abs(nearest - freq) <= thres:
                assign_dict = {
                    "name": molecule,
                    "source": "Scan-{}".format(scan_id),
                    "catalog_frequency": freq,
                    "index": index,
                    "frequency": nearest
                }
                assign_dict.update(**kwargs)
                self._assign_line(**assign_dict)
                counter += 1
            else:
                self.logger.info("No U-line was sufficiently close.")
                self.logger.info("Expected: {}, Nearest: {}".format(freq, nearest))
        self.logger.info("Tentatively assigned {} lines to {}.".format(counter, molecule))

    def process_artifacts(self, frequencies):
        """
        Function for specific for removing anomalies as "artifacts"; e.g. instrumental spurs and
        interference, etc.
        Assignments made this way fall under the "Artifact" category in the resulting assignment
        tables.
         frequencies: list-like with floats corresponding to artifact frequencies
        """
        counter = 0
        self.logger.info("Processing artifacts.")
        for freq in frequencies:
            uline_freqs = np.array([uline.frequency for uline in self.ulines.values()])
            nearest, index = routines.find_nearest(uline_freqs, freq)
            self.logger.info("Found ")
            if np.abs(nearest - freq) <= 0.1:
                assign_dict = {
                    "name": "Artifact",
                    "index": index,
                    "frequency": freq,
                    "catalog_frequency": freq,
                    "source": "Artifact"
                }
                self._assign_line(**assign_dict)
                counter += 1
        self.logger.info("Removed {} lines as artifacts.".format(counter))

    def process_db(self, auto=True, dbpath=None):
        """
        Function for assigning peaks based on a local database file.
        The database is controlled with the SpectralCatalog class, which will handle all of the searching.

        Parameters
        ----------
        auto : bool, optional
            If True, the assignments are made automatically.
        dbpath : str or None, optional
            Filepath to the local database. If none is supplied, uses the default value from the user's home directory.

        """
        old_nulines = len(self.ulines)
        db = database.SpectralCatalog(dbpath)
        self.logger.info("Processing local database: {}".format(dbpath))
        for uindex, uline in tqdm(list(self.ulines.items())):
            self.logger.info("Searching database for {:.4f}".format(uline.frequency))
            catalog_df = db.search_frequency(uline.frequency, self.session.freq_prox, self.session.freq_abs)
            if catalog_df is not None:
                catalog_df["frequency"].replace(0., np.nan, inplace=True)
                catalog_df["frequency"].fillna(catalog_df["catalog_frequency"], inplace=True)
                if len(catalog_df) > 0:
                    sliced_catalog = self.calc_line_weighting(
                        uline.frequency,
                        catalog_df,
                        prox=self.session.freq_prox,
                        abs=self.session.freq_abs,
                        freq_col="frequency"
                    )
                    if self.verbose is True:
                        display(HTML(sliced_catalog.to_html()))
                    if auto is False:
                        index = int(input("Please choose a candidate by index."))
                    elif auto is True:
                        index = 0
                    if index in catalog_df.index:
                        select_df = catalog_df.iloc[index]
                        # Create an approximate quantum number string
                        new_dict = {
                            "index": uindex,
                            "frequency": uline.frequency,
                            "source": "Database",
                            "deviation": uline.frequency - select_df["frequency"]
                        }
                        assign_dict = select_df.to_dict()
                        assign_dict.update(**new_dict)
                        # Use assign_line function to mark peak as assigned
                        self._assign_line(**assign_dict)
                    else:
                        raise IndexError("Invalid index chosen for assignment.")
        # Update the internal table
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        self.logger.info("Prior number of ulines: {}".format(old_nulines))
        self.logger.info("Current number of ulines: {}".format(len(self.ulines)))
        self.logger.info("Finished processing local database.")

    def _assign_line(self, name, index=None, frequency=None, **kwargs):
        """ Mark a transition as assigned, and dump it into
            the assignments list attribute.

            The two methods for doing this is to supply either:
                1. U-line index
                2. U-line frequency
            One way or the other, the U-line Transition object
            will be updated to signify the new assignment.
            Optional kwargs will also be passed to the Transition
            object to update any other details.

            Parameters
            -----------------
             name: str denoting the name of the molecule
             index: optional arg specifying U-line index
             frequency: optional float specifying frequency to assign
             kwargs: passed to update Transition object
        """
        if index == frequency:
            raise Exception("Index/Frequency not specified!")
        ass_obj = None
        if index:
            # If an index is supplied, pull up from uline list
            ass_obj = self.ulines[index]
        elif frequency:
            uline_freqs = np.array([uline.frequency for uline in self.ulines.values()])
            nearest, index = routines.find_nearest(uline_freqs, frequency)
            deviation = np.abs(frequency - nearest)
            # Check that the deviation is at least a kilohertz
            if deviation <= 1E-3:
                self.logger.info("Found U-line number {}.".format(index))
                ass_obj = self.ulines[index]
        if ass_obj:
            ass_obj.name = name
            ass_obj.uline = False
            # Unpack anything else
            ass_obj.__dict__.update(**kwargs)
            if frequency is None:
                frequency = ass_obj.frequency
            ass_obj.frequency = frequency
            self.logger.info("{:,.4f} assigned to {}".format(frequency, name))
            # Delete the line from the ulines dictionary
            del self.ulines[index]
            self.logger.info("Removed U-line index {}.".format(index))
            # Remove from peaks dataframe
            #nearest, array_index = routines.find_nearest(self.peaks["Frequency"].values, frequency)
            #self.logger.info(self.peaks.iloc[array_index])
            #self.peaks.drop(array_index, inplace=True)
            #self.logger.info("Removed {:,.4f}, index {} from peaks table.".format(nearest, array_index))
            self.assignments.append(ass_obj)
        else:
            raise Exception("Peak not found! Try providing an index.")

    def blank_spectrum(self, noise, noise_std, window=1.):
        """
        Blanks a spectrum based on the lines already previously assigned. The required arguments are the average
        and standard deviation of the noise, typically estimated by picking a region free of spectral features.

        The spectra are sequentially blanked - online catalogs first, followed by literature species, finally the
        private assignments.

        Parameters
        ----------
        noise - float
            Average noise value for the spectrum. Typically measured by choosing a region void of spectral lines.
        noise_std - float
            Standard deviation for the spectrum noise.
        window - float
            Value to use for the range to blank. This region blanked corresponds to frequency+/-window.

        Returns
        -------
        """
        sources = ["CDMS/JPL", "Literature", "New"]
        slices = [
            self.table.loc[self.table["source"] == "CDMS/JPL"],
            self.table.loc[(self.table["source"] != "CDMS/JPL") & (self.table["public"] == True)],
            self.table.loc[(self.table["source"] != "CDMS/JPL") & (self.table["public"] == False)]
        ]
        for index, (df, source) in enumerate(zip(slices, sources)):
            try:
                if len(df) > 0:
                    if index == 0:
                        reference = self.int_col
                    else:
                        reference = last_source
                    blanked_spectrum = analysis.blank_spectrum(
                        self.data,
                        df["frequency"].values,
                        noise,
                        noise_std,
                        self.freq_col,
                        reference,
                        window,
                        df=False
                    )
                    self.data[source] = blanked_spectrum
                    last_source = source
            except (KeyError, ValueError):
                self.logger.warning("Could not blank spectrum {} with {}.".format(last_source, source))

    def _get_assigned_names(self):
        """ Method for getting all the unique molecules out
            of the assignments, and tally up the counts.

            :return identifications: dict containing a tally of molecules
                                     identified
        """
        names = [ass_obj.name for ass_obj in self.assignments]
        # Get unique names
        seen = set()
        seen_add = seen.add
        self.names = [name for name in names if not (name in seen or seen_add(name))]
        # Tally up the molecules
        self.identifications = {
            name: names.count(name) for name in self.names
        }
        return self.identifications

    def create_uline_ftb_batch(self, filepath=None, shots=500, dipole=1.):
        """
        Create an FTB file for use in QtFTM based on the remaining ulines. This is used to provide cavity
        frequencies.

        Parameters
        ----------
        shots - int
            Number of shots to integrate on each frequency
        dipole - float
            Dipole moment in Debye attenuation target for each frequency
        """
        if filepath is None:
            filepath = "./ftb/{}-ulines.ftb".format(self.session.experiment)
        lines = ""
        for index, uline in self.ulines.items():
            lines += fa.generate_ftb_line(
                uline.frequency,
                shots,
                **{"dipole": dipole}
            )
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def create_uline_dr_batch(self, filepath=None, select=None, shots=25, dipole=1., gap=500.):
        """
        Create an FTB batch file for use in QtFTM to perform a DR experiment.
        A list of selected frequencies can be used as the cavity frequencies, which will
        subsequently be exhaustively DR'd against by all of the U-line frequencies
        remaining in this experiment.

        Parameters
        ----------
        filepath: str, optional
            Path to save the ftb file to. Defaults to ftb/{}-dr.ftb
        select: list of floats, optional
            List of frequencies to use as cavity frequencies. Defaults to None, which
            will just DR every frequency against each other.
        shots: int, optional
            Number of integration shots
        dipole: float, optional
            Dipole moment used for attenuation setting
        gap: float, optional
            Minimum frequency difference between cavity and DR frequency to actually perform
            the experiment
        """
        if select is None:
            cavity_freqs = [uline.frequency for index, uline in self.ulines]
        else:
            cavity_freqs = select
        dr_freqs = [uline.frequency for index, uline in self.ulines]
        lines = ""
        for cindex, cavity in enumerate(cavity_freqs):
            for dindex, dr in enumerate(dr_freqs):
                if dindex == 0:
                    lines += fa.generate_ftb_line(
                        cavity,
                        shots,
                        **{"dipole": dipole, "drpower": -20}
                    )
                if np.abs(cavity - dr) >= gap:
                    lines += fa.generate_ftb_line(
                        cavity,
                        shots,
                        **{"dipole": dipole, "drpower": 13, "skiptune": True, "drfreq": dr}
                    )
        if filepath is None:
            filepath = "ftb/{}-dr.ftb".format(self.session.experiment)
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def analyze_molecule(self, Q=None, T=None, name=None, formula=None, smiles=None, chi_thres=10.):
        """
        Function for providing some astronomically relevant parameters by analyzing Gaussian line shapes.

        Parameters
        ----------
        Q - float
            Partition function at temperature T
        T - float
            Temperature in Kelvin
        name - str, optional
            Name of the molecule to perform the analysis on. Can be used as a selector.
        formula - str, optional
            Chemical formula of the molecule to perform the analysis on. Can be used as a selector.
        smiles - str, optional
            SMILES code of the molecule to perform the analysis on. Can be used as a selector,
        chi_thres - float
            Threshold for the Chi Squared value to consider fits for statistics. Any instances of fits with Chi
            squared values above this value will not be used to calculate line profile statistics.

        Returns
        -------
        return_data - list
            First element is the profile dataframe, and second element is the fitted velocity.
            If a rotational temperature analysis is also performed, the third element will be the least-squares
            regression.

        """
        if name:
            selector = "name"
            value = name
        elif formula:
            selector = "formula"
            value = formula
        elif smiles:
            selector = "smiles"
            value = smiles
        else:
            raise Exception("No valid selector specified! Please give a name, formula, or SMILES code.")
        # Loop over all of the assignments
        mol_data = list()
        for index, ass_obj in enumerate(self.assignments):
            # If the assignment matches the criteria
            # we perform the analysis
            if ass_obj.__dict__[selector] == value:
                self.logger.info("Performing line profile analysis on assignment index {}.".format(index))
                # Perform a Gaussian fit whilst supplying as much information as we can
                # The width is excluded because it changes significantly between line profiles
                fit_result, summary = analysis.fit_line_profile(
                    self.data,
                    center=ass_obj.frequency,
                    intensity=ass_obj.intensity,
                    freq_col=self.freq_col,
                    int_col=self.int_col,
                    logger=self.logger
                    )
                # If the fit actually converged and worked
                if fit_result:
                    # Calculate what the lab frame frequency would be in order to calculate the frequency offset
                    lab_freq = fit_result.best_values["center"] + units.dop2freq(
                        self.session.velocity, fit_result.best_values["center"]
                    )
                    summary["Frequency offset"] = lab_freq - ass_obj.catalog_frequency
                    summary["Doppler velocity"] = units.freq2vel(
                        ass_obj.catalog_frequency,
                        summary["Frequency offset"]
                    )
                    if Q is not None and T is not None:
                        # Add the profile parameters to list
                        profile_dict = aa.lineprofile_analysis(
                            fit_result,
                                ass_obj.I,
                                Q,
                                T,
                                ass_obj.ustate_energy
                                )
                        ass_obj.N = profile_dict["N cm$^{-2}$"]
                        summary.update(profile_dict)
                    ass_obj.fit = fit_result
                    mol_data.append(summary)
        # If there are successful analyses performed, format the results
        if len(mol_data) > 0:
            profile_df = pd.DataFrame(
                data=mol_data
                )
            # Sort the dataframe by ascending order of chi square - better fits are at the top
            profile_df.sort_values(["Chi squared"], inplace=True)
            # Threshold the dataframe to ensure good statistics
            profile_df = profile_df.loc[profile_df["Chi squared"] <= chi_thres]
            # Calculate the weighted average VLSR based on the goodness-of-fit
            profile_df.loc[:, "Weight"] = profile_df["Chi squared"].max() / profile_df["Chi squared"].values
            weighted_avg = np.sum(profile_df["Weight"] * profile_df["Doppler velocity"]) / np.sum(profile_df["Weight"])
            # Calculate the weighted standard deviation
            stdev = np.sum(profile_df["Weight"] * (profile_df["Doppler velocity"] - weighted_avg)**2) / \
                           np.sum(profile_df["Weight"])
            self.logger.info(
                "Calculated VLSR: {:.3f}+/-{:.3f} based on {} samples.".format(
                    weighted_avg, stdev, len(profile_df)
                )
            )
            return_data = [profile_df, ufloat(weighted_avg, stdev)]
            # If there's data to perform a rotational temperature analysis, then do it!
            if "L" in profile_df.columns:
                self.logger.info("Performing rotational temperature analysis.")
                rot_temp = aa.rotational_temperature_analysis(
                    profile_df["L"],
                    profile_df["E upper"]
                )
                self.logger.info(rot_temp.fit_report())
                return_data.append(rot_temp)
            return return_data
        else:
            self.logger.info("No molecules found, or fits were unsuccessful!")
            return None

    def finalize_assignments(self):
        """
            Function that will complete the assignment process by
            serializing DataClass objects and formatting a report.

            Creates summary pandas dataframes as self.table and self.profiles,
            which correspond to the assignments and fitted line profiles respectively.
        """
        if len(self.assignments) > 0:
            for ass_obj in self.assignments:
                # Dump all the assignments into YAML format
                ass_obj.to_file(
                    "assignment_objs/{}-{}".format(ass_obj.experiment, ass_obj.peak_id),
                    "yaml"
                )
            # Convert all of the assignment data into a CSV file
            ass_df = pd.DataFrame(
                data=[ass_obj.__dict__ for ass_obj in self.assignments]
            )
            self.table = ass_df
            # Dump assignments to disk
            ass_df.to_csv("reports/{0}.csv".format(self.session.experiment), index=False)
            # Update the uline peak list with only unassigned stuff
            try:
                self.peaks = self.peaks[~self.peaks["Frequency"].isin(self.table["frequency"])]
            except KeyError:
                self.logger.warning("Could not compare assignments with peak table - ignoring for now.")
            # Dump Uline data to disk
            self.peaks.to_csv("reports/{0}-ulines.csv".format(self.session.experiment), index=False)

            tally = self._get_assigned_names()
            combined_dict = {
                "assigned_lines": len(self.assignments),
                "ulines": len(self.ulines),
                "peaks": self.peaks[self.freq_col].values,
                "num_peaks": len(self.peaks[self.freq_col]),
                "tally": tally,
                "unique_molecules": self.names,
                "num_unique": len(self.names)
            }
            # Combine Session information
            combined_dict.update(self.session.__dict__)
            # Dump to disk
            routines.dump_yaml(
                "sessions/{0}.yml".format(self.session.experiment),
                "yaml"
            )
            self._create_html_report()
            # Dump data to notebook output
            for key, value in combined_dict.items():
                self.logger.info(key + ":   " + str(value))
        else:
            self.logger.warning("No assignments made in this session - nothing to finalize!")

    def clean_folder(self, action=False):
        """
            Method for cleaning up all of the directories used by this routine.
            Use with caution!!!

            Requires passing a True statement to actually clean up.

            Parameters
            ----------

            action : bool
                If True, folders will be deleted. If False (default) nothing is done.
        """
        folders = ["assignment_objs", "queries", "sessions", "clean", "reports"]
        if action is True:
            for folder in folders:
                rmtree(folder)

    def update_database(self, dbpath=None):
        """
        Adds all of the entries to a specified SpectralCatalog database. The database defaults
        to the global database stored in the home directory. This method will remove everything
        in the database associated with this experiment's ID, and re-add the entries.

        Parameters
        ----------
        dbpath : str, optional
            path to a SpectralCatalog database. Defaults to the system-wide catalog.

        """
        with database.SpectralCatalog(dbpath) as db_obj:
            # Tabula rasa
            db_obj.remove_experiment(self.session.experiment)
            ass_list = [assignment.__dict__ for assignment in self.assignments]
            db_obj.insert_multiple(ass_list)

    def simulate_sticks(self, catalogpath, N, Q, T, doppler=None, gaussian=False):
        """
        Simulates a stick spectrum with intensities in flux units (Jy) for
        a given catalog file, the column density, and the rotational partition
        function at temperature T.

        Parameters
        ----------
         catalogpath : str
            path to SPCAT catalog file
         N : float
            column density in cm^-2
         Q : float
            partition function at temperature T
         T : float
            temperature in Kelvin
         doppler : float, optional
            doppler width in km/s; defaults to session wide value
         gaussian : bool, optional
                   if True, simulates Gaussian profiles instead of sticks
         Returns
         -------
        :return: if gaussian is False, returns a dataframe with sticks; if True,
                 returns a simulated Gaussian line profile spectrum
        """
        # If no Doppler width provided, use the session wide value
        if doppler is None:
            doppler = self.session.doppler
        catalog_df = aa.simulate_catalog(catalogpath, N, Q, T, doppler)
        # Take only the values within band
        catalog_df = catalog_df[
            (catalog_df["Frequency"] >= self.data[self.freq_col].min()) &
            (catalog_df["Frequency"] <= self.data[self.freq_col].max())
        ]
        if gaussian is False:
            return catalog_df[["Frequency", "Flux (Jy)"]]
        else:
            # Convert Doppler width to frequency widths
            widths = units.dop2freq(
                doppler,
                catalog_df["Frequency"].values
            )
            # Calculate the Gaussian amplitude
            amplitudes = catalog_df["Flux (Jy)"] / np.sqrt(2. * np.pi**2. * widths)
            sim_y = self.simulate_spectrum(
                self.data[self.freq_col],
                catalog_df["Frequency"].values,
                widths,
                amplitudes
            )
            simulated_df = pd.DataFrame(
                data=list(zip(self.data[self.freq_col], sim_y)),
                columns=["Frequency", "Flux (Jy)"]
            )
            return simulated_df

    def simulate_spectrum(self, x, centers, widths, amplitudes, fake=False):
        """
            Generate a synthetic spectrum with Gaussians with the
            specified parameters, on a given x axis.

            GaussianModel is used here to remain internally consistent
            with the rest of the code.

             x: array of x values to evaluate Gaussians on
             centers: array of Gaussian centers
             widths: array of Gaussian widths
             amplitudes: array of Gaussian amplitudes
             fake: bool indicating whether false intensities are used for the simulation
            :return y: array of y values
        """
        y = np.zeros(len(x))
        model = GaussianModel()
        for c, w, a in zip(centers, widths, amplitudes):
            if fake is True:
                scaling = a
            else:
                scaling = 1.
            y += scaling * model.eval(
                x=x,
                center=c,
                sigma=w,
                amplitude=a
                )
        return y

    def calculate_assignment_statistics(self):
        """
        Function for calculating some aggregate statistics of the assignments and u-lines. This
        breaks the assignments sources up to identify what the dominant source of information was.
        The two metrics for assignments are the number of transitions and the intensity contribution
        assigned by a particular source.
        :return: dict
        """
        reduced_table = self.table[
            ["frequency", "intensity", "formula",
             "name", "catalog_frequency", "deviation",
             "ustate_energy", "source", "public"]
        ]
        artifacts = reduced_table.loc[reduced_table["name"] == "Artifact"]
        splat = reduced_table.loc[reduced_table["source"] == "CDMS/JPL"]
        local = reduced_table.loc[
            (reduced_table["source"] != "Artifact") &
            (reduced_table["source"] != "CDMS/JPL")
            ]
        public = local.loc[local["public"] == True]
        private = local.loc[local["public"] == False]
        sources = ["Artifacts", "Splatalogue", "Published molecules", "Unpublished molecules"]
        # Added up the total number of lines
        total_lines = len(self.ulines) + len(self.assignments)
        # Add up the total intensity
        total_intensity = np.sum([uline.intensity for uline in self.ulines.values()])
        total_intensity += np.sum(reduced_table["intensity"])
        line_breakdown = [len(source) for source in [artifacts, splat, public, private]]
        intensity_breakdown = [np.sum(source["intensity"]) for source in [artifacts, splat, public, private]]
        # Calculate the aggregate statistics
        cum_line_breakdown = np.cumsum(line_breakdown)
        cum_int_breakdown = np.cumsum(intensity_breakdown)
        # Organize results into dictionary for return
        return_dict = {
            "sources": sources,
            # These are absolute values
            "abs": {
                "total lines": total_lines,
                "total intensity": total_intensity,
                "line breakdown": line_breakdown,
                "intensity breakdown": intensity_breakdown,
                "cumulative line breakdown": cum_line_breakdown,
                "cumulative intensity breakdown": cum_int_breakdown
                },
            # These are the corresponding values in percentage
            "percent": {
                "line breakdown": [(value / total_lines) * 100. for value in line_breakdown],
                "intensity breakdown": [(value / total_intensity) * 100. for value in intensity_breakdown],
                "cumulative line breakdown": [(value / total_lines) * 100. for value in cum_line_breakdown],
                "cumulative intensity breakdown": [(value / total_intensity) * 100. for value in cum_int_breakdown]
            },
            "molecules": {
                "CDMS/JPL": {name: len(splat.loc[splat["name"] == name]) for name in splat["name"].unique()},
                "Published": {name: len(public.loc[public["name"] == name]) for name in public["name"].unique()},
                "Unpublished": {name: len(private.loc[private["name"] == name]) for name in private["name"].unique()}
            }
        }
        return return_dict

    def plot_spectrum(self, simulate=False):
        """
            Generates a Plotly figure of the spectrum. If U-lines are
            present, it will plot the simulated spectrum also.
        """
        fig = go.FigureWidget()

        fig.layout["xaxis"]["title"] = "Frequency (MHz)"
        fig.layout["xaxis"]["tickformat"] = ".,"

        fig.add_scattergl(
            x=self.data[self.freq_col],
            y=self.data[self.int_col],
            name="Experiment",
            opacity=0.6
            )

        if hasattr(self, "ulines"):
            labels = [uline.peak_id for uline in self.ulines.values()]
            amplitudes = np.array([uline.intensity for uline in self.ulines.values()])
            centers = np.array([uline.frequency for uline in self.ulines.values()])
            # Add sticks for U-lines
            fig.add_bar(
                x=centers,
                y=amplitudes,
                hoverinfo="text",
                text=labels,
                name="Peaks"
                )

            if simulate is True:
                widths = units.dop2freq(
                    self.session.doppler,
                    centers
                    )

                simulated = self.simulate_spectrum(
                    self.data[self.freq_col].values,
                    centers,
                    widths,
                    amplitudes,
                    fake=True
                    )

                self.simulated = pd.DataFrame(
                    data=list(zip(self.data[self.freq_col].values, simulated)),
                    columns=["Frequency", "Intensity"]
                    )

                fig.add_scattergl(
                    x=self.simulated["Frequency"],
                    y=self.simulated["Intensity"],
                    name="Simulated spectrum"
                    )

        return fig

    def _create_html_report(self, filepath=None):
        """
        Function for generating an HTML report for sharing. The HTML report is rendered with
        Jinja2, and uses the template "report_template.html" located in the module directory.
        The report includes interactive plots showing statistics of the assignments/ulines and
        an overview of the spectrum. At the end of the report is a table of the assignments and
        uline data.
         filepath: str path to save the report to. Defaults to reports/{id}-summary.html
        """
        from jinja2 import Template
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "report_template.html"
        )
        with open(template_path) as read_file:
            template = Template(read_file.read())
        html_dict = dict()
        # Say what the minimum value for peak detection is.
        html_dict["peak_threshold"] = self.threshold
        # The assigned molecules table
        reduced_table = self.table[
            ["frequency", "intensity", "formula", "name", "catalog_frequency", "deviation", "ustate_energy", "source"]
        ]
        # Render pandas dataframe HTML with bar annotations
        reduced_table_html = reduced_table.style.bar(
            subset=["deviation", "ustate_energy"],
            align="mid",
            color=['#d65f5f', '#5fba7d']
        )\
            .bar(
                subset=["intensity"],
                color="#5fba7d"
            )\
            .format(
                {
                    "frequency": "{:.4f}",
                    "catalog_frequency": "{:.4f}",
                    "deviation": "{:.3f}",
                    "ustate_energy": "{:.2f}",
                    "intensity": "{:.3f}"
                }
            ).set_table_attributes("""class = "durr" id="assignment-table" """)\
            .render(classes=""" "durr" id="assignment-table" """)
        reduced_table_html += """
        <script src="https://code.jquery.com/jquery-1.12.4.js" type="text/javascript"></script>
        <script src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.min.js" type="text/javascript"></script>
        <script type="text/javascript">
        $(document).ready(function() {
          $('#assignment-table').DataTable();
        });
        </script>
        """
        html_dict["assignments_table"] = reduced_table_html
        # The unidentified features table
        uline_df = pd.DataFrame(
            [[uline.frequency, uline.intensity] for uline in self.ulines.values()], columns=["Frequency", "Intensity"]
        )
        html_dict["uline_table"] = uline_df.style.bar(
            subset=["Intensity"],
            color='#5fba7d'
        )\
            .format(
                {
                    "Frequency": "{:.4f}",
                    "Intensity": "{:.2f}"
                }
            ).render()
        # Plotly displays of the spectral feature breakdown and whatnot
        html_dict["plotly_breakdown"] = plot(self.plot_breakdown(), output_type="div")
        html_dict["plotly_figure"] = plot(self.plot_assigned(), output_type="div")
        # Render the template with Jinja and save the HTML report
        output = template.render(session=self.session, **html_dict)
        if filepath is None:
            filepath = "reports/{}-summary.html".format(self.session.experiment)
        with open(filepath, "w+") as write_file:
            write_file.write(output)

    def plot_breakdown(self):
        """
        Generate two charts to summarize the breakdown of spectral features.
        The left column plot shows the number of ulines being assigned by
        the various sources of frequency data.

        Artifacts - instrumental interference, from the function `process_artifacts`
        Splatalogue - uses the astroquery API, from the function `splat_assign_spectrum`
        Published - local catalogs, but with the `public` kwarg flagged as True
        Unpublished - local catalogs, but with the `public` kwarg flagged as False
        :return: Plotly FigureWidget object
        """
        fig = figurefactory.init_plotly_subplot(
            nrows=1, ncols=2,
            **{"subplot_titles": ("Reference Breakdown", "Intensity Breakdown")}
        )
        fig.layout["title"] = "Spectral Feature Breakdown"
        fig.layout["showlegend"] = False
        reduced_table = self.table[
            ["frequency", "intensity", "formula",
             "name", "catalog_frequency", "deviation",
             "ustate_energy", "source", "public"]
        ]
        artifacts = reduced_table.loc[reduced_table["name"] == "Artifact"]
        splat = reduced_table.loc[reduced_table["source"] == "CDMS/JPL"]
        local = reduced_table.loc[
            (reduced_table["source"] != "Artifact") &
            (reduced_table["source"] != "CDMS/JPL")
        ]
        public = local.loc[local["public"] == True]
        private = local.loc[local["public"] == False]
        sources = ["Artifacts", "Splatalogue", "Published molecules", "Unpublished molecules"]
        # Added up the total number of lines
        total_lines = len(self.ulines) + len(self.assignments)
        # Add up the total intensity
        total_intensity = np.sum([uline.intensity for uline in self.ulines.values()])
        total_intensity += np.sum(reduced_table["intensity"])
        line_breakdown = [total_lines]
        intensity_breakdown = [total_intensity]
        for source in [artifacts, splat, public, private]:
            line_breakdown.append(-len(source))
            intensity_breakdown.append(-np.sum(source["intensity"]))
        line_breakdown = np.cumsum(line_breakdown)
        intensity_breakdown = np.cumsum(intensity_breakdown)
        labels = ["Total"] + sources
        colors = ["#d7191c", "#fdae61", "#ffffbf", "#abdda4", "#2b83ba"]
        # Left column plot of the number of lines assigned
        fig.add_trace(
            go.Scattergl(
                x=labels,
                y=line_breakdown,
                fill="tozeroy",
                hoverinfo="x+y"
            ),
            1,
            1
        )
        # Bar charts showing the number of lines from each source
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[0.] + [len(source) for source in [artifacts, splat, public, private]],
                hoverinfo="x+y",
                width=0.5,
                marker={"color": colors}
            ),
            1,
            1
        )
        # Right column plot of the intensity contributions
        fig.add_trace(
            go.Scattergl(
                x=labels,
                y=intensity_breakdown,
                fill="tozeroy",
                hoverinfo="x+y"
            ),
            1,
            2
        )
        # Bar chart showing the intensity contribution from each source
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[0.] + [np.sum(source["intensity"]) for source in [artifacts, splat, public, private]],
                hoverinfo="x+y",
                width=0.5,
                marker={"color": colors}
            ),
            1,
            2
        )
        fig["layout"]["xaxis1"].update(
            title="Source",
            showgrid=True
        )
        fig["layout"]["yaxis1"].update(
            title="Cumulative number of assignments",
            range=[0., total_lines * 1.05]
        )
        fig["layout"]["xaxis2"].update(
            title="Source",
            showgrid=True
        )
        fig["layout"]["yaxis2"].update(
            title="Cumulative intensity",
            range=[0., total_intensity * 1.05]
        )
        return fig

    def plot_assigned(self):
        """
            Generates a Plotly figure with the assignments overlaid
            on the experimental spectrum.

            Does not require any parameters, but requires that
            the assignments and peak finding functions have been
            run previously.
        """
        fig = go.FigureWidget()
        fig.layout["title"] = "Experiment {}".format(self.session.experiment)
        fig.layout["xaxis"]["title"] = "Frequency (MHz)"
        fig.layout["xaxis"]["tickformat"] = ".2f"

        # Update the peaks table
        self.peaks = pd.DataFrame(
            data=[[uline.frequency, uline.intensity] for uline in self.ulines.values()],
            columns=["Frequency", "Intensity"]
        )

        fig.add_scattergl(
            x=self.data["Frequency"],
            y=self.data["Intensity"],
            name="Experiment",
            opacity=0.6
        )

        fig.add_bar(
            x=self.table["catalog_frequency"],
            y=self.table["intensity"],
            width=1.0,
            hoverinfo="text",
            text=self.table["name"].astype(str) + "-" + self.table["r_qnos"].astype(str),
            name="Assignments"
        )
        ulines = np.array([[uline.intensity, uline.frequency] for index, uline in self.ulines.items()])

        fig.add_bar(
            x=ulines[:,1],
            y=ulines[:,0],
            width=1.0,
            name="U-lines"
        )
        return fig

    def stacked_plot(self, frequencies, freq_range=0.05):
        """
        Special implementation of the stacked_plot from the figurefactory module, adapted
        for AssignmentSession. In this version, the assigned/u-lines are also indicated.

        This function will generate a Plotly figure that stacks up the spectra as subplots,
        with increasing frequencies going up the plot. This function was written primarily
        to identify harmonically related lines, which in the absence of centrifugal distortion
        should line up perfectly in the center of the plot.

        Due to limitations with Plotly, there is a maximum of ~8 plots that can stacked
        and will return an Exception if > 8 frequencies are provided.

         frequencies: list of floats, corresponding to center frequencies
         freq_range: float percentage value of each center frequency to use as cutoffs
        :return: Plotly Figure object
        """
        # Update the peaks table
        self.peaks = pd.DataFrame(
            data=[[uline.frequency, uline.intensity] for uline in self.ulines.values()],
            columns=["Frequency", "Intensity"]
        )
        dataframe = self.data.copy()
        # Want the frequencies in ascending order, going upwards in the plot
        indices = np.where(
            np.logical_and(
                dataframe[self.freq_col].min() <= frequencies,
                frequencies <= dataframe[self.freq_col].max()
            )
        )
        # Plot only frequencies within band
        frequencies = frequencies[indices]
        # Sort frequencies such that plots are descending in frequency
        frequencies = np.sort(frequencies)[::-1]
        nplots = len(frequencies)

        titles = tuple("{:.0f} MHz".format(frequency) for frequency in frequencies)
        fig = figurefactory.init_plotly_subplot(
            nrows=nplots, ncols=1,
            **{
                "subplot_titles": titles,
                "vertical_spacing": 0.15,
                "shared_xaxes": True
            }
        )
        for index, frequency in enumerate(frequencies):
            # Calculate the offset frequency
            dataframe["Offset " + str(index)] = dataframe[self.freq_col] - frequency
            # Range as a fraction of the center frequency
            freq_cutoff = freq_range * frequency
            max_freq = frequency + freq_cutoff
            min_freq = frequency - freq_cutoff
            sliced_df = dataframe.loc[
                (dataframe["Offset " + str(index)] > -freq_cutoff) & (dataframe["Offset " + str(index)] < freq_cutoff)
                ]
            sliced_peaks = self.peaks.loc[
                (self.peaks["Frequency"] <= max_freq) & (min_freq <= self.peaks["Frequency"])
            ]
            sliced_peaks["Offset Frequency"] = sliced_peaks["Frequency"] - frequency
            sliced_assignments = self.table.loc[
                (self.table["frequency"] <= max_freq) & (min_freq <= self.table["frequency"])
            ]
            sliced_assignments["offset_frequency"] = sliced_assignments["catalog_frequency"] - frequency
            # Plot the data
            traces = list()
            # Spectrum plot
            traces.append(
                    go.Scattergl(
                    x=sliced_df["Offset " + str(index)],
                    y=sliced_df[self.int_col],
                    mode="lines",
                    opacity=0.6,
                    marker={"color": "rgb(5,113,176)"}
                )
            )
            traces.append(
                go.Bar(
                    x=sliced_assignments["offset_frequency"],
                    y=sliced_assignments["intensity"],
                    width=1.0,
                    hoverinfo="text",
                    text=sliced_assignments["name"] + "-" + sliced_assignments["r_qnos"],
                    name="Assignments",
                    marker={"color": "rgb(253,174,97)"}
                )
            )
            traces.append(
                go.Bar(
                    x=sliced_peaks["Offset Frequency"],
                    y=sliced_peaks["Intensity"],
                    width=1.0,
                    name="U-lines",
                    marker={"color": "rgb(26,150,65)"},
                    hoverinfo="text",
                    text=sliced_peaks["Frequency"]
                )
            )
            # Plotly indexes from one because they're stupid
            fig.add_traces(traces, [index + 1] * 3, [1] * 3)
            fig["layout"]["xaxis1"].update(
                range=[-freq_cutoff, freq_cutoff],
                title="Offset frequency (MHz)",
                showgrid=True
            )
            fig["layout"]["yaxis" + str(index + 1)].update(showgrid=False)
        fig["layout"].update(
            autosize=True,
            height=1000,
            width=900,
            showlegend=False
        )
        return fig

    def match_artifacts(self, artifact_exp, threshold=0.05):
        """
        Remove artifacts based on another experiment which has the blank
        sample - i.e. only artifacts.

        The routine will simple match peaks found in the artifact
        experiment, and assign all coincidences in the present experiment
        as artifacts.

        Parameters
        ----------
        artifact_exp - AssignmentSession object
            Experiment with no sample present
        threshold - float, optional
            Threshold in absolute frequency units for matching
        """
        matches = analysis.match_artifacts(self, artifact_exp, threshold)
        self.process_artifacts([freq for index, freq in matches.items()])
        for index, freq in matches.items():
            self.logger.info("Removed {} peak as artifact.".format(freq))

    def find_progressions(
            self, search=0.001, low_B=400.,
            high_B=9000., sil_calc=True, refit=False, plot=True, **kwargs
    ):
        """
            High level function for searching U-line database for
            possible harmonic progressions. Wraps the lower level function
            harmonic_search, which will generate 3-4 frequency combinations
            to search.

            Parameters
            ---------------
             search - threshold for determining if quantum number J is close
                     enough to an integer/half-integer
             low_B - minimum value for B
             high_B - maximum value for B
             plot - whether or not to produce a plot of the progressions

            Returns
            ---------------
             harmonic_df - dataframe containing the viable transitions
        """
        uline_frequencies = [uline.frequency for uline in self.ulines.values()]
        progressions = analysis.harmonic_finder(
            uline_frequencies,
            search=search,
            low_B=low_B,
            high_B=high_B
        )
        fit_df = fitting.harmonic_fitter(progressions, J_thres=search)[0]
        self.harmonic_fits = fit_df
        # Run cluster analysis on the results
        self.cluster_dict, self.progressions, self.cluster_obj = analysis.cluster_AP_analysis(
            self.harmonic_fits,
            sil_calc,
            refit,
            **kwargs
        )
        self.cluster_df = pd.DataFrame.from_dict(self.cluster_dict).T
        return self.cluster_df

    def search_species(self, formula=None, name=None, smiles=None):
        """
            Method for finding species in the assigned dataframe,
            with the intention of showing where the observed frequencies
            are.

            Parameters
            --------------
            formula - str for chemical formula lookup
            name - str for common name
            smiles - str for unique SMILES string

            Returns
            --------------
            pandas dataframe slice with corresponding lookup
        """
        if hasattr(self, "table") is False:
            raise Exception("No assignment table created yet. Finalize assignments.")
        if formula:
            locator = "formula"
            check = formula
        if name:
            locator = "name"
            check = name
        if smiles:
            locator = "smiles"
            check = smiles
        return self.table.loc[self.table[locator] == check]

    def save_session(self, filepath=None):
        """
            Method to save an AssignmentSession to disk.
            
            The underlying mechanics are based on the joblib library,
            and so there can be cross-compatibility issues particularly
            when loading from different versions of Python.

            Parameters
            ---------------
             filepath - str
                Path to save the file to. By default it will go into the sessions folder.
        """
        if filepath is None:
            filepath = "./sessions/{}.pkl".format(self.session.experiment)
        self.logger.info("Saving session to {}".format(filepath))
        if hasattr(self, "log_handlers"):
            del self.log_handlers
        # Save to disk
        routines.save_obj(
            self,
            filepath
        )

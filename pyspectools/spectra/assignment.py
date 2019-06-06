
"""
    `assignment` module

    This module contains three main classes for performing analysis of broad-
    band spectra. The `AssignmentSession` class will be what the user will
    mainly interact with, which will digest a spectrum, find peaks, make
    assignments and keep track of them, and generate the reports at the end.

    To perform the assignments, the user can use the `LineList` class, which
    does the grunt work of homogenizing the different sources of frequency
    and molecular information: it is able to take SPCAT and .lin formats, as
    well as simply a list of frequencies. `LineList` then interacts with the
    `AssignmentSession` class, which handles the assignments.

    The smallest building block in this procedure is the `Transition` class;
    every peak, every molecule transition, every artifact is considered as
    a `Transition` object. The `LineList` contains a list of `Transition`s,
    and the peaks found by the `AssignmentSession` are also kept as a 
    `LineList`.

"""

import os
from shutil import rmtree
from dataclasses import dataclass, field
from typing import List, Dict
from copy import copy, deepcopy
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
from lmfit.models import GaussianModel
from IPython.display import display, HTML
from periodictable import formula
from plotly.offline import plot
from plotly import graph_objs as go
from uncertainties import ufloat
from jinja2 import Template

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
    Data class for handling parameters used for an AssignmentSession.
    The user generally shouldn't need to directly interact with this class,
    but can give some level of dynamic control and bookkeeping to how and
    what molecules can be assigned, particularly with the composition, the
    frequency thresholds for matching, and the noise statistics.

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
    """
        Main class for bookkeeping and analyzing broadband spectra. This class
        revolves around operating on a single continuous spectrum, using the
        class functions to automatically assess the noise statistics, find
        peaks, and do the bulk of the bookkeeping on what molecules are assigned
        to what peak.

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
            col_names=None, freq_col="Frequency", int_col="Intensity", skiprows=0, verbose=False, **kwargs
    ):
        """
        Class method for AssignmentSession to generate a session using an ASCII
        file. This is the preferred method for starting an AssignmentSession.
        The ASCII files are parsed using the pandas method `read_csv`, with
        the arguments for reading simply passed to that function.

        Example based on blackchirp spectrum:
        The first row in an ASCII output from blackchirp contains the headers,
        which typically should be renamed to "Frequency, Intensity". This can
        be done with this call:

        ```
        session = AssignmentSession.from_ascii(
            filepath="ft1020.txt",
            experiment=0,
            col_names=["Frequency", "Intensity"],
            skiprows=1
            )
        ```

        Example based on astronomical spectra:
        File formats are not homogenized, and delimiters may change. This exam-
        ple reads in a comma-separated spectrum, with a radial velocity of
        +26.2 km/s.
        
        ```
        session = AssignmentSession.from_ascii(
            filepath="spectrum.mid.dat",
            experiment=0,
            col_names=["Frequency", "Intensity"],
            velocity=26.2,
            delimiter=","
            )
        ```

        Parameters
        ----------
        filepath : str
            Filepath to the ASCII spectrum
        experiment : int
            Integer identifier for the experiment
        composition : list of str, optional
            List of atomic symbols, representing the atomic composition of the experiment
        delimiter : str, optional
            Delimiter character used in the ASCII file. For example, "\t", "\s", ","
        velocity: float, optional
            Radial velocity to offset the frequency in km/s.
        temperature : float, optional
            Rotational temperature in Kelvin used for the experiment.
        col_names : None or list of str, optional
            Names to rename the columns. If None, this is ignored.
        freq_col : str, optional
            Name of the column to be used for the frequency axis
        int_col : str, optional
            Name of the column to be used for the intensity axis
        skip_rows : int, optional
            Number of rows to skip reading.
        verbose : bool, optional
            If True, the logging module will also print statements and display
            any interaction that happens.
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
        if hasattr(self, "log_handlers"):
            for key, handler in self.log_handlers.items():
                handler.close()

    def __init__(
            self, exp_dataframe, experiment, composition, temperature=4.0, velocity=0.,
            freq_col="Frequency", int_col="Intensity", verbose=True, **kwargs
    ):
        """ init method for AssignmentSession.

            Attributes
            ----------
            session : `Session` object
                Class containing parameters for the experiment, including the
                chemical composition, the noise statistics, radial velocity,
                etc.
            data : Pandas DataFrame
                This pandas dataframe contains the actual x/y data for the
                spectrum being analyzed.
            freq_col, int_col : str
                Names of the frequency and intensity columns that are contained
                in the `self.data` dataframe
            t_threshold : float
                This value is used to cut off upper-states for assignment. This
                corresponds to three times the user specified temperature for
                the experiment.
            umols : list
                TODO this list should be used to keep track of unidentified
                molecules during the assignment process. Later on if an the
                carrier is identified we should be able to update everything
                consistently.
            verbose : bool
                Specifies whether the logging is printed in addition to being
                dumped to file.
            line_lists : dict
                Dictionary containing all of the `LineList` objects being used
                for assignments. When the `find_peaks` function is run, a
                `LineList` is generated, holding every peak found as a corres-
                ponding `Transition`. This `LineList` is then referenced by
                the "Peaks" key in the line_lists dictionary.
        """
        # Make folders for organizing output
        folders = [
            "assignment_objs", "queries", "sessions", "clean", "figures",
            "reports", "logs", "outputs", "ftb", "linelists"
        ]
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
        # Initial threshold for peak detection is set to None
        self.threshold = None
        self.umols = list()
        self.verbose = verbose
        # Holds catalogs
        self.line_lists = dict()
        self._init_logging()
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
        This gives the ratio of spectral intensities, and can be useful for determining
        whether a line becomes stronger or weaker between experiments.

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
        # If verbose is specified, the logging info is directly printed as well
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

    def find_peaks(self, threshold=None, region=None, sigma=6, min_dist=10):
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
            min_dist=min_dist
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
        self.logger.info("Modifying the experiment LineList.")
        # Set up session information to be passed in the U-line
        skip = [
            "temperature", "doppler", "freq_abs", "freq_prox", "noise_rms",
            "baseline", "header", "noise_region", "composition", "name"
        ]
        selected_session = {
            key: self.session.__dict__[key] for key in self.session.__dict__ if key not in skip
            }
        # If the Peaks key has not been set up yet, we set it up now
        if "Peaks" not in self.line_lists:
            self.line_lists["Peaks"] = LineList.from_dataframe(
                dataframe, name="Peaks",
                freq_col=freq_col, int_col=int_col, **selected_session
            )
        # Otherwise, we'll just update the existing LineList
        else:
            vfunc = np.vectorize(Transition)
            transitions = vfunc(
                frequency=dataframe[freq_col],
                intensity=dataframe[int_col],
                **selected_session
            )
            self.line_lists["Peaks"].update_linelist(transitions)
        self.logger.info("There are now {} line entries in this session.".format(len(self.line_lists["Peaks"])))
        peaks_df = self.line_lists["Peaks"].to_dataframe()[
            ["frequency", "intensity"]
        ]
        peaks_df.columns = ["Frequency", "Intensity"]
        self.peaks = peaks_df

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

    def apply_filter(self, window, sigma=0.5, int_col=None):
        """
        Applies a filter to the spectral signal. If multiple window functions
        are to be used, a list of windows can be provided, which will then
        perform the convolution stepwise. With the exception of the gaussian
        window function, the others functions use the SciPy signal functions.

        A reference copy of the original signal is kept as the "Ref" column;
        this is used if a new window function is applied, rather than on the
        already convolved signal.

        Parameters
        ----------
        window: str, or iterable of str
            Name of the window function
        sigma: float, optional
            Specifies the magnitude of the gaussian blur. Only used when the
            window function asked for is "gaussian".
        int_col: None or str, optional
            Specifies which column to apply the window function to. If None,
            defaults to the session-wide intensity column
        """
        if int_col is None:
            int_col = self.int_col
        # If the reference spectrum exists, we'll use that instead
        if "Ref" in self.data.columns:
            int_col = "Ref"
            self.logger.info("Using Ref signal for window function.")
        else:
            # Make a copy of the original signal
            self.data["Ref"] = self.data[self.int_col].copy()
            self.logger.info("Copied signal to Ref column.")
        intensity = self.data[int_col].values
        self.logger.info(
            "Applying {} to column {}.".format(window, int_col)
        )
        # Try to see if the window variable is an iterable
        try:
            assert type(window) == list
            for function in iter(window):
                intensity = analysis.filter_spectrum(
                    intensity, function, sigma
                )
        except AssertionError:
            # In the event that window is not a list, just apply the filter
            intensity = analysis.filter_spectrum(
                intensity, window, sigma
            )
        # Windowed signal is usually too small for float printing in the html
        # report
        self.data[int_col] = intensity * 1000.

    def splat_assign_spectrum(self, auto=False):
        """
        Alias for `process_splatalogue`. Function will be removed in a later version.

        Parameters
        ----------
        auto : bool
            Specifies whether the assignment procedure is automatic.
        """
        self.process_splatalogue(auto=auto)

    def process_splatalogue(self, auto=True, progressbar=True):
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
        ulines = self.line_lists["Peaks"].get_ulines()
        self.logger.info("Beginning Splatalogue lookup on {} lines.".format(len(ulines)))
        iterator = enumerate(ulines)
        if progressbar is True:
            iterator = tqdm(iterator)
        for index, uline in iterator:
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

                splat_df = analysis.calc_line_weighting(
                    frequency, splat_df, prox=self.session.freq_prox, abs=self.session.freq_abs
                )
                if splat_df is not None:
                    self.logger.info("Found {} candidates for frequency {:,.4f}, index {}.".format(
                        len(splat_df), frequency, index)
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
                            "queries/{0}-{1}.csv".format(self.session.experiment, index), index=False
                        )
                        ass_dict = {
                            "uline": False,
                            "frequency": frequency,
                            "intensity": uline.intensity,
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
                        # Update the Transition entry
                        uline.__dict__.update(**ass_dict)
                    except ValueError:
                        # If nothing matches, keep in the U-line
                        # pile.
                        self.logger.info("Deferring assignment for index {}.".format(index))
                else:
                    # Throw into U-line pile if no matches at all
                    self.logger.info("No species known for {:,.4f}".format(frequency))
        self.logger.info("Splatalogue search finished.")

    def process_linelist(self, name=None, formula=None, filepath=None,
                         linelist=None, auto=True, thres=-10.,
                         progressbar=True, tol=None, **kwargs,):
        """
        General purpose function for performing line assignments using local catalog and line data. The two main ways
        of running this function is to either provide a linelist or filepath argument. The type of linelist will be
        checked to determine how the catalog data will be processed: if it's a string, it will be used to use


        Parameters
        ----------
        name: str, optional
            Name of the molecule being assigned. This should be specified when providing a new line list, which then
            gets added to the experiment.
        formula: str, optional
            Chemical formula for the molecule being assigned. Should be added in conjuction with name.
        filepath: str, optional
            If a linelist is not given, a filepath can be specified corresponding to a .cat or .lin file, which will
            be used to create a LineList object.
        linelist: str or LineList, optional
            Can be the name of a molecule or LineList object; the former is specified as a string which looks up the
            experiment line_list attribute for an existing LineList object. If a LineList object is provided, the
            function will use this directly.
        auto: bool, optional
            Specifies whether the assignment procedure works without intervention. If False, the user will be prompted
            to provide a candidate index.
        thres: float, optional
            log Intensity cut off used to screen candidates.
        progressbar: bool, optional
            If True, a tqdm progressbar will indicate assignment progress.
        tol: float, optional
            Tolerance for making assignments. If None, the function will default to the session-wide values of
            freq_abs and freq_prox to determine the tolerance.
        kwargs
            Kwargs are passed to the Transition object update when assignments are made.
        """
        # First case is if linelist is provided as a string corresponding to the key in the line_lists attribute
        self.logger.info("Processing local catalog for molecule {}.".format(name))
        if type(linelist) == str:
            linelist = self.line_lists[linelist]
        # In the event that linelist is an actual LineList object
        elif type(linelist) == LineList:
            if linelist.name not in self.line_lists:
                self.line_lists[linelist.name] = linelist
        elif filepath:
            # If a catalog is specified, create a LineList object with this catalog. The parser is inferred from the
            # file extension.
            if ".cat" in filepath:
                func = LineList.from_catalog
            elif ".lin" in filepath:
                func = LineList.from_lin
            else:
                raise Exception("File extension for reference line list not recognized!")
            linelist = func(
                name=name, formula=formula, filepath=filepath,
                min_freq=self.data[self.freq_col].min(), max_freq=self.data[self.freq_col].max()
            )
            if name not in self.line_lists:
                self.line_lists[name] = linelist
        else:
            raise Exception("Please specify an internal or external line list!")
        if linelist is not None:
            nassigned = 0
            iterator = enumerate(self.line_lists["Peaks"].get_ulines())
            if progressbar is True:
                iterator = tqdm(iterator)
            # Loop over all of the U-lines
            for index, transition in iterator:
                # If no value of tolerance is provided, determine from the session
                if tol is None:
                    if self.session.freq_abs is True:
                        tol = self.session.freq_prox
                    else:
                        tol = (1. - self.session.freq_prox) * transition.frequency
                self.logger.info(
                    "Searching for frequency {} with tolerances: {} K, +/-{} MHz, {} intensity.".format(
                        transition.frequency,
                        self.t_threshold,
                        tol,
                        thres
                    )
                )
                # Find transitions in the LineList that can match
                can_pkg = linelist.find_candidates(
                    transition.frequency,
                    lstate_threshold=self.t_threshold,
                    freq_tol=tol,
                    int_tol=thres
                )
                # If there are actual candidates instead of NoneType, we can process it.
                if can_pkg is not None:
                    candidates, weighting = can_pkg
                    ncandidates = len(candidates)
                    self.logger.info("Found {} possible matches.".format(ncandidates))
                    # If auto mode or if there's just one candidate, just take the highest weighting
                    if auto is True or ncandidates == 1:
                        chosen = candidates[weighting.argmax()]
                    else:
                        for cand_idx, candidate in enumerate(candidates):
                            print(cand_idx, candidate)
                        chosen_idx = int(
                            input("Please specify the candidate index.   ")
                        )
                        chosen = candidates[chosen_idx]
                    self.logger.info(
                        "Assigning {} (catalog {:.4f}) to peak {} at {:.4f}.".format(
                            chosen.name, chosen.catalog_frequency, index, transition.frequency
                        )
                    )
                    # Create a copy of the Transition data from the LineList
                    assign_dict = copy(chosen.__dict__)
                    # Update with the measured frequency and intensity
                    assign_dict["frequency"] = transition.frequency
                    assign_dict["intensity"] = transition.intensity
                    assign_dict["velocity"] = self.session.velocity
                    assign_dict["uline"] = False
                    # Add any other additional kwargs
                    assign_dict.update(
                        **kwargs
                    )
                    # Copy over the information from the assignment, and update
                    # the experimental peak information with the assignment
                    transition.__dict__.update(**assign_dict)
                    nassigned += 1
            self.logger.info(
                "Assigned {} new transitions to {}.".format(
                    nassigned, linelist.name
                )
            )
        else:
            self.logger.warning("LineList was empty, and no lines were assigned.")

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
                    sliced_catalog = analysis.line_weighting(
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
        names = [ass_obj.name for ass_obj in self.line_lists["Peaks"].get_assignments()]
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
        filepath: str or None, optional
            Path to save the .ftb file to. If None, defaults to the session ID.
        shots: int
            Number of shots to integrate on each frequency
        dipole: float
            Dipole moment in Debye attenuation target for each frequency
        """
        if filepath is None:
            filepath = "./ftb/{}-ulines.ftb".format(self.session.experiment)
        lines = ""
        for index, uline in enumerate(self.line_lists["Peaks"].get_ulines()):
            lines += fa.generate_ftb_line(
                uline.frequency,
                shots,
                **{"dipole": dipole}
            )
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def create_uline_dr_batch(self, filepath=None, select=None,
                              shots=25, dipole=1., gap=500., thres=None):
        """
        Create an FTB batch file for use in QtFTM to perform a DR experiment.
        A list of selected frequencies can be used as the cavity frequencies, which will
        subsequently be exhaustively DR'd against by all of the U-line frequencies
        remaining in this experiment.

        The file is then saved to "ftb/XXX-dr.ftb".

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
        thres: None or float, optional
            Minimum value in absolute intensity units to consider in the DR
            batch. If None, this is ignored (default).
        """
        ulines = self.line_lists["Peaks"].get_ulines()
        if select is None:
            cavity_freqs = [uline.frequency for uline in ulines]
        else:
            cavity_freqs = select
        dr_freqs = [uline.frequency for uline in ulines]
        if thres is not None:
            intensities = np.array([uline.intensity for uline in ulines])
            mask = np.where(intensities >= thres)
            dr_freqs = np.asarray(dr_freqs)[mask]
            cavity_freqs = np.asarray(cavity_freqs)
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
        assignments = self.line_lists["Peaks"].get_assignments()
        ulines = self.line_lists["Peaks"].get_ulines()
        if len(assignments) > 0:
            for obj in assignments:
                # Dump all the assignments into YAML format
                obj.to_file(
                    "assignment_objs/{}-{}".format(obj.experiment, obj.peak_id),
                    "yaml"
                )
                obj.deviation = obj.catalog_frequency - obj.frequency
            # Convert all of the assignment data into a CSV file
            assignment_df = pd.DataFrame(
                data=[obj.__dict__ for obj in assignments]
            )
            self.table = assignment_df
            # Dump assignments to disk
            assignment_df.to_csv(
                "reports/{0}.csv".format(self.session.experiment),
                index=False
            )
            # Dump Uline data to disk
            peak_data = [
                [peak.frequency, peak.intensity] for peak in ulines
            ]
            peak_df = pd.DataFrame(
                peak_data, columns=["Frequency", "Intensity"]
            )
            peak_df.to_csv(
                "reports/{0}-ulines.csv".format(self.session.experiment),
                index=False
            )

            tally = self._get_assigned_names()
            combined_dict = {
                "assigned_lines": len(assignments),
                "ulines": len(ulines),
                "peaks": self.line_lists["Peaks"].get_frequencies(),
                "num_peaks": len(self.line_lists["Peaks"]),
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
        total_lines = len(self.line_lists["Peaks"])
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

        if "Peaks" in self.line_lists:
            ulines = self.line_lists["Peaks"].get_ulines()
            labels = list(range(len(ulines)))
            amplitudes = np.array([uline.intensity for uline in ulines])
            centers = np.array([uline.frequency for uline in ulines])
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
            ["frequency", "intensity", "formula", "name", "catalog_frequency",
             "deviation", "ustate_energy", "source"]
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
        ulines = self.line_lists["Peaks"].get_ulines()
        uline_df = pd.DataFrame(
            [[uline.frequency, uline.intensity] for uline in ulines], columns=["Frequency", "Intensity"]
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
        ulines = self.line_lists["Peaks"].get_ulines()
        assignments = self.line_lists["Peaks"].get_assignments()
        total_lines = len(ulines) + len(assignments)
        # Add up the total intensity
        total_intensity = np.sum([uline.intensity for uline in ulines])
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
            data=[[uline.frequency, uline.intensity] for uline in self.line_lists["Peaks"].get_ulines()],
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
        ulines = np.array([[uline.intensity, uline.frequency] for uline in self.line_lists["Peaks"].get_ulines()])

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
        ulines = self.line_lists["Peaks"].get_ulines()
        self.peaks = pd.DataFrame(
            data=[[uline.frequency, uline.intensity] for uline in ulines],
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
        frequencies = np.asarray(frequencies)[indices]
        # Sort frequencies such that plots are descending in frequency
        frequencies = np.sort(frequencies)[::-1]
        nplots = len(frequencies)
        if nplots > 5:
            raise Exception(
                "Too many requested frequencies; I can't stack them all!"
            )
        titles = tuple(
            "{:.0f} MHz".format(frequency) for frequency in frequencies
        )
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
                dataframe["Offset {}".format(index)].between(
                    -freq_cutoff, freq_cutoff
                )
            ]
            sliced_peaks = self.peaks.loc[
                self.peaks["Frequency"].between(min_freq, max_freq)
            ]
            sliced_peaks["Offset Frequency"] = sliced_peaks["Frequency"] - frequency
            sliced_assignments = self.table.loc[
                self.table["frequency"].between(min_freq, max_freq)
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
                    text=sliced_assignments["name"] + "-" + sliced_assignments["r_qnos"].apply(str),
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
            self, search=0.001, low_B=400., high_B=9000.,
            refit=False, plot=True, preferences=None, **kwargs
    ):
        """
        Performs a search for possible harmonically related U-lines.
        The first step loops over every possible U-line pair, and uses the
        difference to estimate an effective B value for predicting the next
        transition. If the search is successful, the U-line is added to the
        list. The search is repeated until there are no more U-lines within
        frequency range of the next predicted line.

        Once the possible series are identified, the frequencies are fit to
        an effective linear molecule model (B and D terms). An affinity propa-
        gation cluster model is then used to group similar progressions toge-
        ther, with either a systematic test of preference values or a user
        specified value.

        Parameters
        ----------
        search: float, optional
            Percentage value of the target frequency cutoff for excluding
            possible candidates in the harmonic search
        low_B, high_B: float, optional
            Minimum and maximum value of B in MHz to be considered. This
            constrains the size of the molecule you are looking for.
        refit: bool, optional
            If True, B and D are refit to the cluster frequencies.
        plot: bool, optional
            If True, creates a Plotly scatter plot of the clusters, as a funct-
            ion of the preference values.
        preferences: float or array_like of floats, optional
            A single value or an array of preference values for the AP cluster
            model. If None, the clustering will be performed on a default grid,
            where all of the results are returned.
        kwargs: optional
            Additional kwargs are passed to the AP model initialization.

        Returns
        -------
        fig
        """
        ulines = self.line_lists["Peaks"].get_ulines()
        uline_frequencies = [uline.frequency for uline in ulines]
        self.logger.info("Performing harmonic progression search")
        progressions = analysis.harmonic_finder(
            uline_frequencies,
            search=search,
            low_B=low_B,
            high_B=high_B
        )
        self.logger.info("Fitting progressions.")
        fit_df, _ = fitting.harmonic_fitter(progressions, J_thres=search)
        self.logger.info("Found {} possible progressions.".format(len(fit_df)))
        pref_test_data = dict()
        # Run cluster analysis on the results
        if preferences is None:
            preferences = np.linspace(-5000., 5000., 10)
        if type(preferences) == list or type(preferences) == np.ndarray:
            for preference in preferences:
                try:
                    ap_settings = {"preference": preference}
                    ap_settings.update(**kwargs)
                    cluster_dict, progressions, _ = analysis.cluster_AP_analysis(
                        fit_df,
                        sil_calc=True,
                        refit=refit,
                        **ap_settings
                    )
                    pref_test_data[preference] = {
                        "cluster_data": cluster_dict,
                        "nclusters": len(cluster_dict),
                        "prog_df": progressions,
                    }
                except ValueError:
                    pass
        else:
            # Perform the AP clustering with a single preference value
            cluster_dict, progressions, _ = analysis.cluster_AP_analysis(
                fit_df,
                sil_calc=True,
                refit=refit,
                **{"preference": preferences}
            )
            pref_test_data[preferences] = {
                "cluster_data": cluster_dict,
                "nclusters":    len(cluster_dict),
                "prog_df":      progressions,
            }
        # Make a plotly figure of how the clustering goes (with frequency)
        # as a function of preference
        if plot is True:
            fig = go.FigureWidget()
            fig.layout["xaxis"]["title"] = "Frequency (MHz)"
            fig.layout["xaxis"]["tickformat"] = ".,"
            for preference, contents in pref_test_data.items():
                df = contents["prog_df"]
                # Create the colors for the unique clusters
                colors = figurefactory.generate_colors(
                    len(contents["cluster_data"]),
                    hex=True,
                    cmap=plt.cm.Spectral
                )
                # Assign a color to each cluster
                cmap = {
                    index: color for index, color in zip(
                        df["Cluster indices"].unique(),
                        colors
                    )
                }
                for index, data in contents["cluster_data"].items():
                    frequencies = data["Frequencies"]
                    fig.add_scattergl(
                        x=frequencies,
                        y=[preference + index] * len(frequencies),
                        mode="markers",
                        marker={"color": cmap[index]},
                        opacity=0.7
                    )
            return fig, pref_test_data
        else:
            return None, pref_test_data

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


@dataclass
class LineList:
    """
        Class for handling and homogenizing all of the possible line lists: from peaks to assignments to catalog files.

        Attributes
        ----------
        name: str, optional
            Name of the line list. Can be used to identify the molecule, or to simply state the purpose of the list.
        formula: str, optional
            Chemical formula for the molecule, if applicable.
        smi: str, optional
            SMILES representation of the molecule, if applicable.
        filecontents: str, optional
            String representation of the file contents used to make the line list.
        filepath: str, optional
            Path to the file used to make the list.
        transitions: list, optional
            A designated list for holding Transition objects. This is the bulk of the information for a given
            line list.
    """
    name: str = ""
    formula: str = ""
    smi: str = ""
    filecontents: str = ""
    filepath: str = ""
    transitions: List = field(default_factory=list)
    frequencies: List[float] = field(default_factory=list)
    catalog_frequencies: List[float] = field(default_factory=list)

    def __str__(self):
        return "Line list for: {}, Formula: {}, Number of entries: {}".format(
            self.name, self.formula, len(self.transitions)
        )

    def __repr__(self):
        return "Line list name: {}, Number of entries: {}".format(
            self.name, len(self.transitions)
        )

    def __len__(self):
        return len(self.transitions)

    def __post_init__(self):
        if len(self.transitions) != 0:
            self.frequencies = [obj.frequency for obj in self.transitions]
            self.catalog_frequencies = [obj.catalog_frequency for obj in self.transitions]

    @classmethod
    def from_catalog(cls, name, formula, filepath, min_freq=0., max_freq=1e12, **kwargs):
        """
        Create a Line List object from an SPCAT catalog.
        Parameters
        ----------
        name: str
            Name of the molecule the catalog belongs to
        formula: str
            Chemical formula of the molecule
        filepath: str
            Path to the catalog file.
        min_freq: float, optional
            Minimum frequency in MHz for the frequency cutoff
        max_freq: float, optional
            Maximum frequency in MHz for the frequency cutoff
        kwargs: optional
            Additional attributes that are passed into the Transition objects.

        Returns
        -------
        linelist_obj
            Instance of LineList with the digested catalog.
        """
        catalog_df = parsers.parse_cat(filepath, low_freq=min_freq, high_freq=max_freq)
        try:
            columns = ["N'", "J'", "N''", "J''"]
            catalog_df["qno"] = catalog_df[columns].apply(
                lambda x: "N'={}, J'={} - N''={}, J''={}".format(*x),
                axis=1
            )
            # Create a formatted quantum number string
            #catalog_df["qno"] = "N'={}, J'={} - N''={}, J''={}".format(
            #    *catalog_df[["N'", "J'", "N''", "J''"]].values
            #)
            # Calculate E upper to have a complete set of data
            catalog_df["Upper state energy"] = units.calc_E_upper(
                catalog_df["Frequency"], catalog_df["Lower state energy"]
            )
            vfunc = np.vectorize(Transition)
            # Vectorized generation of all the Transition objects
            transitions = vfunc(
                catalog_frequency=catalog_df["Frequency"],
                catalog_intensity=catalog_df["Intensity"],
                lstate_energy=catalog_df["Lower state energy"],
                ustate_energy=catalog_df["Upper state energy"],
                r_qnos=catalog_df["qno"],
                source="Catalog",
                name=name,
                formula=formula,
                uline=False,
                **kwargs
            )
            linelist_obj = cls(name, formula, filepath=filepath, transitions=list(transitions))
            return linelist_obj
        except IndexError:
            return None

    @classmethod
    def from_dataframe(cls, dataframe, name="Peaks",
                       freq_col="Frequency", int_col="Intensity", **kwargs
                       ):
        """
        Specialized class method for creating a LineList object from a Pandas Dataframe. This method is called by
        the AssignmentSession.df2ulines function to generate a Peaks LineList during peak detection.

        Parameters
        ----------
        dataframe: pandas DataFrame
            DataFrame containing frequency and intensity information
        freq_col: str, optional
            Name of the frequency column
        int_col: str, optional
            Name of the intensity column
        kwargs
            Optional settings are passed into the creation of Transition objects.

        Returns
        -------
        LineList
        """
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            frequency=dataframe[freq_col],
            intensity=dataframe[int_col],
            uline=True,
            **kwargs
        )
        linelist_obj = cls(name=name, transitions=list(transitions))
        return linelist_obj

    @classmethod
    def from_lin(cls, name, linpath, formula="", **kwargs):
        """
        Generate a LineList object from a .lin file. This method should be used for intermediate assignments, when one
        does not know what the identity of a molecule is but has measured some frequency data.

        Parameters
        ----------
        name: str
            Name of the molecule
        linpath: str
            File path to the .lin file.
        formula: str, optional
            Chemical formula of the molecule if known.
        kwargs
            Additional kwargs are passed into the Transition objects.

        Returns
        -------
        LineList
        """
        lin_df = parsers.parse_lin(linpath)
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            name=name,
            formula=formula,
            catalog_frequency=lin_df["Frequency"],
            r_qnos=lin_df["Quantum numbers"],
            uline=False,
            source="Line file",
            **kwargs
        )
        linelist_obj = cls(name, formula, filepath=linpath, transitions=list(transitions))
        return linelist_obj

    @classmethod
    def from_artifacts(cls, frequencies, **kwargs):
        """
        Specialized class method for creating a LineList object specifically for artifacts/RFI. These Transitions are
        specially flagged as Artifacts.

        Parameters
        ----------
        frequencies: iterable of floats
            List or array of floats corresponding to known artifact frequencies.
        kwargs
            Kwargs are passed into the Transition object creation.

        Returns
        -------
        LineList
        """
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            name="Artifact",
            catalog_frequency=np.asarray(frequencies),
            uline=False,
            source="Artifact",
            **kwargs
        )
        linelist_obj = cls(name="Artifacts", transitions=list(transitions))
        return linelist_obj

    def to_dataframe(self):
        """
        Convert the transition data into a Pandas DataFrame.
        Returns
        -------
        dataframe
            Pandas Dataframe with all of the transitions in the line list.
        """
        list_rep = [obj.__dict__ for obj in self.transitions]
        return pd.DataFrame(list_rep)

    def to_pickle(self, filepath=None):
        """
        Function to serialize the LineList to a Pickle file. If no filepath is provided, the function will default
        to using the name attribute of the LineList to name the file.

        Parameters
        ----------
        filepath: str or None, optional
            If None, uses name attribute for the filename, and saves to the linelists folder.
        """
        if filepath is None:
            filepath = "linelists/{}-linelist.pkl".format(self.name)
        routines.save_obj(self, filepath)

    def find_nearest(self, frequency, tol=1e-3):
        """
        Look up transitions to find the nearest in frequency to the query. If the matched frequency is within a
        tolerance, then the function will return the corresponding Transition. Otherwise, it returns None.

        Parameters
        ----------
        frequency: float
            Frequency in MHz to search for.
        tol: float, optional
            Maximum tolerance for the deviation from the LineList frequency and query frequency

        Returns
        -------
        Transition object or None
        """
        match_freq, index = routines.find_nearest(self.frequencies, frequency)
        deviation = np.abs(frequency - match_freq)
        if deviation <= tol:
            return self.transitions[index]
        else:
            return None

    def find_candidates(self, frequency, lstate_threshold=4.,
                        freq_tol=1e-1, int_tol=-10.):
        """
        Function for searching the LineList for candidates. The first step uses pure Python to isolate transitions
        that meet three criteria: the lower state energy, the catalog intensity, and the frequency distance.

        If no candidates are found, the function will return None. Otherwise, it will return the list of transitions
        and a list of associated normalized weights.

        Parameters
        ----------
        frequency: float
            Frequency in MHz to try and match.
        lstate_threshold: float, optional
            Lower state energy threshold in Kelvin
        freq_tol: float, optional
            Frequency tolerance in MHz for matching two frequencies
        int_tol: float, optional
            log Intensity threshold

        Returns
        -------
        transitions, weighting or None
            If candidates are found, lists of the transitions and the associated weights are returned.
            Otherwise, returns None
        """
        # first threshold the temperature
        transitions = [
            obj for obj in self.transitions if all(
                [
                    obj.lstate_energy <= lstate_threshold,
                    obj.catalog_intensity >= int_tol,
                    abs(obj.catalog_frequency - frequency) <= freq_tol]
            )
        ]
        # If there are candidates, calculate the weights associated with each transition
        if len(transitions) != 0:
            transition_frequencies = np.array(
                [obj.catalog_frequency for obj in transitions]
            )
            transition_intensities = np.array(
                [obj.catalog_intensity for obj in transitions]
            )
            # If there are actually no catalog intensities, it should sum up to zero in which case we won't
            # use the intensities in the weight factors
            if np.sum(transition_intensities) == 0.:
                transition_intensities = None
            weighting = analysis.line_weighting(
                frequency, transition_frequencies, transition_intensities
            )
            # Only normalize if there's more than one
            if len(weighting) > 1:
                weighting /= weighting.max()
            return transitions, weighting
        else:
            return None

    def update_transition(self, index, **kwargs):
        """
        Function for updating a specific Transition object within the Line List.

        Parameters
        ----------
        index: int
            Index for the list Transition object
        kwargs: optional
            Updates to the Transition object
        """
        self.transitions[index].__dict__.update(**kwargs)

    def update_linelist(self, transition_objs):
        """
        Adds transitions to a LineList if they do not exist in the list already.

        Parameters
        ----------
        transition_objs: list
            List of Transition objects
        """
        self.transitions.extend(
            [obj for obj in transition_objs if obj not in self.transitions]
        )

    def get_ulines(self):
        """
        Function for retrieving unidentified lines in a Line List.

        Returns
        -------
        uline_objs: list
            List of all of the transition objects where the uline flag is set to True.
        """
        uline_objs = [obj for obj in self.transitions if obj.uline is True]
        return uline_objs

    def get_assignments(self):
        """
        Function for retrieving assigned lines in a Line List.

        Returns
        -------
        assign_objs: list
            List of all of the transition objects where the uline flag is set to False.
        """
        assign_objs = [obj for obj in self.transitions if obj.uline is False]
        return assign_objs

    def get_frequencies(self):
        return [transition.frequency for transition in self.transitions]

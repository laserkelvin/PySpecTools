"""
    assignment.py

    Contains dataclass routines for tracking assignments
    in broadband spectra.
"""

import os
from shutil import rmtree
from dataclasses import dataclass, field
from typing import List, Dict
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from lmfit.models import GaussianModel
from IPython.display import display, HTML
from periodictable import formula
from plotly.offline import plot
from plotly import graph_objs as go

from pyspectools import routines, parsers, figurefactory
from pyspectools import fitting
from pyspectools import units
from pyspectools import database
from pyspectools.astro import analysis as aa
from pyspectools.spectra import analysis


@dataclass
class Assignment:
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
        I : float
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
        intereference : bool
            Flag to indicate if this assignment is not molecular in nature
        source : str
            Indicates what the source used for this assignment is
        public : bool
            Flag to indicate if the information for this assignment is public/published
    """
    name: str = ""
    smiles: str = ""
    formula: str = ""
    frequency: float = 0.0
    catalog_frequency: float = 0.0
    catalog_intensity: float = 0.0
    deviation: float = 0.0
    intensity: float = 0.0
    I: float = 0.0
    peak_id: int = 0
    experiment: int = 0
    uline: bool = True
    composition: List[str] = field(default_factory=list)
    v_qnos: List[int] = field(default_factory=list)
    r_qnos: str = ""
    fit: Dict = field(default_factory=dict)
    ustate_energy: float = 0.0
    interference: bool = False
    weighting: float = 0.0
    source: str = "Catalog"
    public: bool = True

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
        Dunder method for representing an Assignment, which returns
        the name of the line and the frequency.

        Returns
        -------
        str
            name and frequency of the Assignment
        """
        return f"{self.name}, {self.frequency}"

    def to_file(self, filepath, format="yaml"):
        """
        Save an Assignment object to disk with a specified file format.
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
        y : Numpy 1D array
            Values of the synthetic Gaussian spectrum at each particular value of x
        """
        model = GaussianModel()
        params = model.make_params()
        params.update(self.fit)
        y = model.eval(params, x=x)
        return y

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
        Assignment
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
        Assignment
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
        Assignment
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
    rv : float
        Radial velocity of the source in km/s; used to offset the frequency spectrum
     freq_prox : float
        frequency cutoff for line assignments. If freq_abs attribute is True, this value is taken as the absolute value.
         Otherwise, it is a percentage of the frequency being compared.
     freq_abs : bool
        If True, freq_prox attribute is taken as the absolute value of frequency, otherwise as a decimal percentage of
         the frequency being compared.
    """
    experiment: int
    composition: List[str] = field(default_factory=list)
    temperature: float = 4.0
    doppler: float = 0.01
    velocity: float = 0.
    freq_prox: float = 0.1
    freq_abs: bool = True

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
        return session

    @classmethod
    def from_ascii(
            cls, filepath, experiment, composition=["C", "H"], delimiter="\t", temperature=4.0, velocity=0.,
            col_names=None, freq_col="Frequency", int_col="Intensity", skiprows=0, **kwargs
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
        session = cls(spec_df, experiment, composition, temperature, velocity, freq_col, int_col, **kwargs)
        return session

    def __init__(
            self, exp_dataframe, experiment, composition, temperature=4.0, velocity=0.,
            freq_col="Frequency", int_col="Intensity", **kwargs
    ):
        """ init method for AssignmentSession.

            Required arguments are necessary metadata for controlling various aspects of
            the automated assignment procedure, as well as for reproducibility.

            Parameters
            -------------------------
             exp_dataframe: pandas dataframe with observational data in frequency/intensity
             experiment: int ID for the experiment
             composition: list of str corresponding to elemental composition composition; e.g. ["C", "H"]
             freq_col: optional str arg specifying the name for the frequency column
             int_col: optional str arg specifying the name of the intensity column
        """
        # Make folders for organizing output
        folders = ["assignment_objs", "queries", "sessions", "clean", "figures", "reports"]
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
        self.data.loc[:, self.freq_col] += doppler_offset

    def find_peaks(self, threshold=None):
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

            Returns
            -------
            peaks_df : dataframe
                Pandas dataframe with Frequency/Intensity columns, corresponding to peaks
        """
        if threshold is None:
            # Set the threshold as 20% of the baseline + 1sigma. The peak_find function will
            # automatically weed out the rest.
            threshold = (self.data[self.int_col].mean() + self.data[self.int_col].std() * 1.5)
            self.threshold = threshold
            print("Peak detection threshold is: {}".format(threshold))
        else:
            self.threshold = threshold
        peaks_df = analysis.peak_find(
            self.data,
            freq_col=self.freq_col,
            int_col=self.int_col,
            thres=threshold,
        )
        # Reindex the peaks
        peaks_df.reset_index(drop=True, inplace=True)
        if hasattr(self, "peaks") is True:
            # If we've looked for peaks previously
            # we don't have to re-add the U-line to
            # the list
            peaks_df = pd.concat([peaks_df, self.peaks])
            # drop repeated frequencies
            peaks_df.drop_duplicates(["Frequency"], inplace=True)
        # Generate U-lines
        skip = ["temperature", "doppler", "freq_abs", "freq_prox"]
        selected_session = {
            key: self.session.__dict__[key] for key in self.session.__dict__ if key not in skip
            }
        for index, row in peaks_df.iterrows():
            ass_obj = Assignment(
                frequency=row[self.freq_col],
                intensity=row[self.int_col],
                peak_id=index,
                **selected_session
            )
            # If the frequency hasn't already been done, add it
            # to the U-line pile
            if ass_obj not in self.ulines.values() and ass_obj not in self.assignments:
                self.ulines[index] = ass_obj
        # Assign attribute
        self.peaks = peaks_df
        return peaks_df

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
        lower_freq = frequency * (1. - self.session.freq_prox)
        upper_freq = frequency * (1 + self.session.freq_prox)
        if hasattr(self, "table"):
            slice_df = self.table.loc[
                (self.table["frequency"] >= lower_freq) &
                (self.table["frequency"] <= upper_freq)
                ]
        # If no hits turn up, look for it in U-lines
        if len(slice_df) == 0:
            print("No assignment found; searching U-lines")
            slice_df = self.peaks.loc[
                (self.peaks["Frequency"] >= lower_freq) &
                (self.peaks["Frequency"] <= upper_freq)
                ]
            if len(slice_df) == 0:
                raise Exception("Frequency not found in U-lines.")
            else:
                print("Found U-lines.")
                return slice_df
        else:
            print("Found assignments.")
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
            throw it into an Assignment object and flag it as known.
            Conversely, if it's not known in Splatalogue it will defer
            assignment, flagging it as unassigned and dumping it into
            the `uline` attribute.

        Parameters
        ----------
         auto: bool
             If True the assignment process does not require user input, otherwise will prompt user.
        """
        if hasattr(self, "peaks") is False:
            print("Peak detection not run; running with default settings.")
            self.find_peaks()

        for uindex, uline in tqdm(list(self.ulines.items())):
            frequency = uline.frequency
            print("Searching for frequency {:,.4f}".format(frequency))
            # Call splatalogue API to search for frequency
            splat_df = analysis.search_center_frequency(frequency, width=0.1)
            # Filter out lines that are way too unlikely on grounds of temperature
            splat_df = splat_df.loc[splat_df["E_U (K)"] <= self.t_threshold]
            # Filter out quack elemental compositions
            for index, row in splat_df.iterrows():
                # Convert the string into a chemical formula object
                try:
                    clean_formula = row["Species"].split("v=")[0]
                    for prefix in ["l-", "c-"]:
                        clean_formula = clean_formula.replace(prefix, "")
                    formula_obj = formula(clean_formula)
                    # Check if proposed molecule contains atoms not
                    # expected in composition
                    comp_check = all(
                        str(atom) in self.session.composition for atom in formula_obj.atoms
                    )
                    if comp_check is False:
                        # If there are crazy things in the mix, forget about it
                        print("Molecule " + clean_formula + " rejected.")
                        splat_df.drop(index, inplace=True)
                except:
                    print("Could not parse molecule " + clean_formula + " rejected.")
                    splat_df.drop(index, inplace=True)
            nitems = len(splat_df)

            splat_df = self.calc_line_weighting(
                frequency, splat_df, prox=self.session.freq_prox, abs=self.session.freq_abs
            )
            if splat_df is not None:
                display(HTML(splat_df.to_html()))
                try:
                    print("Observed frequency is {:,.4f}".format(frequency))
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
                        "ustate_energy": ass_df["E_U (K)"][0],
                        "weighting": ass_df["Weighting"][0],
                        "source": "CDMS/JPL",
                        "deviation": frequency - ass_df["Frequency"][0]
                    }
                    # Perform a Voigt profile fit
                    print("Attempting to fit line profile...")
                    fit_results = analysis.fit_line_profile(
                        self.data,
                        frequency,

                    )
                    # Pass the fitted parameters into Assignment object
                    ass_dict["fit"] = fit_results.best_values
                    # Need support to convert common name to SMILES
                    self.assign_line(**ass_dict)
                except ValueError:
                    # If nothing matches, keep in the U-line
                    # pile.
                    print("Deferring assignment")
            else:
                # Throw into U-line pile if no matches at all
                print("No species known for {:,.4f}".format(frequency))
            display(HTML("<hr>"))
        print("Processed {} lines.".format(uindex))

    def process_lin(self, name, formula, linpath, auto=True, **kwargs):
        """
            Reads in a line file and sweeps through the U-line list.

            Operationally, the same as the catalog and splatalogue methods,
            but parses a line file instead. The differences are a lack of
            filtering, since there is insufficient information in a lin
            file.

            Kwargs are passed to the `assign_line` function, which provides
            the user with additional options for flagging an Assignment
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

        for uindex, uline in tqdm(list(self.ulines.items())):
            sliced_catalog = self.calc_line_weighting(
                uline.frequency,
                lin_df,
                prox=self.session.freq_prox,
                abs=self.session.freq_abs
            )
            if sliced_catalog is not None:
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
                        "index": uline.peak_id,
                        "frequency": uline.frequency,
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "source": "Line file",
                        "deviation": uline.frequency - select_df["Frequency"]
                    }
                    assign_dict.update(**kwargs)
                    self.assign_line(**assign_dict)
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        print("Prior number of ulines: {}".format(old_nulines))
        print("Current number of ulines: {}".format(len(self.ulines)))

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
            (catalog_df[freq_col] >= lower_freq) & (catalog_df[freq_col] <= upper_freq)
            ]
        nentries = len(sliced_catalog)
        if nentries > 0:
            # Calculate probability weighting. Base is the inverse of distance
            sliced_catalog.loc[:, "Deviation"] = np.abs(sliced_catalog[freq_col] - frequency)
            sliced_catalog.loc[:, "Weighting"] = (1. / sliced_catalog["Deviation"])
            # If intensity is included in the catalog incorporate it in
            # the weight calculation
            if int_col in sliced_catalog:
                sliced_catalog.loc[:, "Weighting"] *= (10 ** sliced_catalog[int_col])
            elif "CDMS/JPL Intensity" in sliced_catalog:
                sliced_catalog.loc[:, "Weighting"] *= (10 ** sliced_catalog["CDMS/JPL Intensity"])
            else:
                # If there are no recognized intensity columns, pass
                pass
            # Normalize the weights
            sliced_catalog.loc[:, "Weighting"] /= sliced_catalog["Weighting"].max()
            # Sort by obs-calc
            sliced_catalog.sort_values(["Weighting"], ascending=False, inplace=True)
            sliced_catalog.reset_index(drop=True, inplace=True)
            return sliced_catalog
        else:
            return None

    def process_catalog(self, name, formula, catalogpath, auto=True, thres=-10., **kwargs):
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
            the Assignment object.

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
        for uindex, uline in tqdm(list(self.ulines.items())):
            # 0.1% of frequency
            sliced_catalog = self.calc_line_weighting(
                uline.frequency, catalog_df, prox=self.session.freq_prox, abs=self.session.freq_abs
            )
            if sliced_catalog is not None:
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
                        "index": uline.peak_id,
                        "frequency": uline.frequency,
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "catalog_intensity": select_df["Intensity"],
                        "source": "Catalog",
                        "deviation": uline.frequency - select_df["Frequency"]
                    }
                    # Pass whatever extra stuff
                    assign_dict.update(**kwargs)
                    # Use assign_line function to mark peak as assigned
                    self.assign_line(**assign_dict)
                else:
                    raise IndexError("Invalid index chosen for assignment.")
        # Update the internal table
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        print("Prior number of ulines: {}".format(old_nulines))
        print("Current number of ulines: {}".format(len(self.ulines)))

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
                self.assign_line(**assign_dict)
                counter += 1
            else:
                print("No U-line was sufficiently close.")
                print("Expected: {}, Nearest: {}".format(freq, nearest))
        print("Tentatively assigned {} lines to {}.".format(counter, molecule))

    def process_artifacts(self, frequencies):
        """
        Function for specific for removing anomalies as "artifacts"; e.g. instrumental spurs and
        interference, etc.
        Assignments made this way fall under the "Artifact" category in the resulting assignment
        tables.
         frequencies: list-like with floats corresponding to artifact frequencies
        """
        counter = 0
        for freq in frequencies:
            uline_freqs = np.array([uline.frequency for uline in self.ulines.values()])
            nearest, index = routines.find_nearest(uline_freqs, freq)
            if np.abs(nearest - freq) <= 0.1:
                assign_dict = {
                    "name": "Artifact",
                    "index": index,
                    "frequency": freq,
                    "catalog_frequency": freq,
                    "source": "Artifact"
                }
                self.assign_line(**assign_dict)
                counter += 1
        print("Removed {} lines as artifacts.".format(counter))

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
        for uindex, uline in tqdm(list(self.ulines.items())):
            print("Searching database for {:.4f}".format(uline.frequency))
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
                    display(HTML(sliced_catalog.to_html()))
                    if auto is False:
                        index = int(input("Please choose a candidate by index."))
                    elif auto is True:
                        index = 0
                    if index in catalog_df.index:
                        select_df = catalog_df.iloc[index]
                        # Create an approximate quantum number string
                        new_dict = {
                            "index": uline.peak_id,
                            "frequency": uline.frequency,
                            "source": "Database",
                            "deviation": uline.frequency - select_df["frequency"]
                        }
                        assign_dict = select_df.to_dict()
                        assign_dict.update(**new_dict)
                        # Use assign_line function to mark peak as assigned
                        self.assign_line(**assign_dict)
                    else:
                        raise IndexError("Invalid index chosen for assignment.")
        # Update the internal table
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
        )
        self.table = ass_df
        print("Prior number of ulines: {}".format(old_nulines))
        print("Current number of ulines: {}".format(len(self.ulines)))

    def assign_line(self, name, index=None, frequency=None, **kwargs):
        """ Mark a transition as assigned, and dump it into
            the assignments list attribute.

            The two methods for doing this is to supply either:
                1. U-line index
                2. U-line frequency
            One way or the other, the U-line Assignment object
            will be updated to signify the new assignment.
            Optional kwargs will also be passed to the Assignment
            object to update any other details.

            Parameters
            -----------------
             name: str denoting the name of the molecule
             index: optional arg specifying U-line index
             frequency: optional float specifying frequency to assign
             kwargs: passed to update Assignment object
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
                print("Found U-line number {}.".format(index))
                ass_obj = self.ulines[index]
        if ass_obj:
            ass_obj.name = name
            ass_obj.uline = False
            # Unpack anything else
            ass_obj.__dict__.update(**kwargs)
            if frequency is None:
                frequency = ass_obj.frequency
            ass_obj.frequency = frequency
            print("{:,.4f} assigned to {}".format(frequency, name))
            # Delete the line from the ulines dictionary
            del self.ulines[index]
            self.assignments.append(ass_obj)
        else:
            raise Exception("Peak not found! Try providing an index.")

    def get_assigned_names(self):
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

    def analyze_molecule(self, Q, T, name=None, formula=None, smiles=None):
        """
            Function for providing some astronomically relevant
            parameters by analyzing Gaussian line shapes.

             Q: rotational partition function at temperature
             T: temperature in K
             name: str name of molecule
             formula: chemical formula of molecule
             smiles: SMILES string for molecule
            :return profile_df: pandas dataframe containing all of the
                                analysis
        """
        profile_data = list()
        if name:
            selector = "name"
            value = name
        if formula:
            selector = "formula"
            value = formula
        if smiles:
            selector = "smiles"
            value = smiles
        # Loop over all of the assignments
        for ass_obj in self.assignments:
            # If the assignment matches the criteria
            # we perform the analysis
            if ass_obj.__dict__[selector] == value:
                # Get width estimate
                width = units.dop2freq(
                    self.session.doppler,
                    ass_obj.frequency
                    )
                # Perform a Gaussian fit
                fit_report = analysis.fit_line_profile(
                    self.data,
                    ass_obj.frequency,
                    width,
                    ass_obj.intensity,
                    freq_col=self.freq_col,
                    int_col=self.int_col
                    )
                # Add the profile parameters to list
                profile_dict = aa.lineprofile_analysis(
                        fit_report,
                        ass_obj.I,
                        Q,
                        T,
                        ass_obj.ustate_energy
                        )
                profile_data.append(profile_dict)
                ass_obj.fit = fit_report
                ass_obj.N = profile_dict["N cm$^{-2}$"]
        if len(profile_data) > 0:
            profile_df = pd.DataFrame(
                data=profile_data
                )
            sim_y = self.simulate_spectrum(
                self.data[self.freq_col],
                profile_df["frequency"].values,
                profile_df["width"].values,
                profile_df["amplitude"].values
                )
            sim_df = pd.DataFrame(
                data=list(zip(
                    self.data[self.freq_col].values,
                    sim_y
                    ))
                )
            return profile_df, sim_df
        else:
            print("No molecules found!")
            return None

    def finalize_assignments(self):
        """
            Function that will complete the assignment process by
            serializing DataClass objects and formatting a report.

            Creates summary pandas dataframes as self.table and self.profiles,
            which correspond to the assignments and fitted line profiles respectively.
        """
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
        self.peaks = self.peaks[~self.peaks["Frequency"].isin(self.table["frequency"])]
        # Dump Uline data to disk
        self.peaks.to_csv("reports/{0}-ulines.csv".format(self.session.experiment), index=False)

        tally = self.get_assigned_names()
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
        self.create_html_report()
        # Dump data to notebook output
        for key, value in combined_dict.items():
            print(key + ":   " + str(value))
        #self.update_database()

    def clean_folder(self, action=False):
        """
            Method for cleaning up all of the directories used by this routine.
            Use with caution!!!

            Requires passing a True statement to actually clean up.

             action: bool; will only clean up when True is passed
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

    def create_html_report(self, filepath=None):
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
        html_dict["assignments_table"] = reduced_table.style.bar(
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
            ).render()
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
            text=self.table["name"] + "-" + self.table["r_qnos"],
            name="Assignments"
        )

        fig.add_bar(
            x=self.peaks["Frequency"],
            y=self.peaks["Intensity"],
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
        self.cluster_dict, self.cluster_obj = analysis.cluster_AP_analysis(
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
        # Save to disk
        routines.save_obj(
            self,
            filepath
        )
        print("Saved session to {}".format(filepath))

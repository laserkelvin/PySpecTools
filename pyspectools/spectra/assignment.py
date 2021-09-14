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
from typing import List, Dict, Tuple, Union, Type, Any
from copy import copy, deepcopy
from itertools import combinations
import warnings
import logging
from pathlib import Path

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
from monsterurl import get_monster
from scipy.ndimage.filters import gaussian_filter

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
        discharge : bool
            Whether or not the line is discharge dependent
        magnet : bool
            Whether or not the line is magnet dependent (i.e. open shell)
    """

    name: str = ""
    smiles: str = ""
    formula: str = ""
    frequency: float = 0.0
    catalog_frequency: float = 0.0
    catalog_intensity: float = 0.0
    deviation: float = 0.0
    intensity: float = 0.0
    uncertainty: float = 0.0
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
    velocity: float = 0.0
    discharge: bool = False
    magnet: bool = False
    multiple: List[str] = field(default_factory=list)
    final: bool = False

    def __eq__(self, other: object) -> bool:
        """ Dunder method for comparing molecules.
            This method is simply a shortcut to see if
            two molecules are the same based on their
            SMILES code, the chemical name, and frequency.
        """
        if not isinstance(other, Transition):
            return NotImplemented
        else:
            comparisons = [
                self.smiles == other.smiles,
                self.name == other.name,
                self.frequency == other.frequency,
                self.v_qnos == other.v_qnos,
            ]
            return all(comparisons)

    def __str__(self):
        """
        Dunder method for representing an Transition, which returns
        the name of the line and the frequency.

        Returns
        -------
        str
            name and frequency of the Transition
        """
        if self.uline is True:
            frequency = self.frequency
        else:
            frequency = self.catalog_frequency
        return f"{self.name}, {frequency:,.4f}"

    def __repr__(self):
        if self.uline is True:
            frequency = self.frequency
        else:
            frequency = self.catalog_frequency
        return f"{self.name}, {frequency:,.4f}"

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __gt__(self, other):
        return self.frequency > other.frequency

    def check_molecule(self, other):
        """
        
        Check equivalency based on a common carrier. Compares the
        name, formula, and smiles of this `Transition` object with
        another.
        
        Returns
        -------
        bool
            True if the two `Transitions` belong to the same carrier.
        """
        assert type(other) == type(self)
        return all(
            [
                self.name == other.name,
                self.formula == other.formula,
                self.smiles == other.smiles,
            ]
        )

    def calc_intensity(self, Q: float, T=300.0):
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
            T,
        )
        self.S = I
        return I

    def calc_linestrength(self, Q: float, T=300.0):
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
            self.S, Q, frequency, units.calc_E_lower(frequency, self.ustate_energy), T
        )
        self.intensity = intensity
        return intensity

    def to_file(self, filepath: str, format="yaml"):
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

    def get_spectrum(self, x: np.ndarray):
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
    def from_dict(cls, data_dict: Dict):
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
        assignment_obj = cls(**data_dict)
        return assignment_obj

    @classmethod
    def from_yml(cls, yaml_path: str):
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
        assignment_obj = cls(**yaml_dict)
        return assignment_obj

    @classmethod
    def from_json(cls, json_path: str):
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
        assignment_obj = cls(**json_dict)
        return assignment_obj

    def reset_assignment(self):
        """
        Function to reset an assigned line into its original state.
        The only information that is kept regards to the frequency,
        intensity, and aspects about the experiment.
        """
        remain = {
            "frequency": self.frequency,
            "intensity": self.intensity,
            "experiment": self.experiment,
            "velocity": self.velocity,
            "source": self.source,
        }
        empty = Transition()
        self.__dict__.update(**empty.__dict__)
        self.__dict__.update(**remain)

    def choose_assignment(self, index: int):
        """
        Function to manually pick an assignment from
        a list of multiple possible assignments found
        during `process_linelist`. After the new assignment
        is copied over, the `final` attribute is set to
        True and will no longer throw a warning duiring
        finalize_assignments.
        
        Parameters
        ----------
        index : int
            Index of the candidate to use for the assignment.
        """
        assert len(self.multiple != 0)
        assert index < len(self.multiple)
        remain = {
            "frequency": self.frequency,
            "intensity": self.intensity,
            "experiment": self.experiment,
            "velocity": self.velocity,
            "source": self.source,
            "multiple": self.multiple,
        }
        chosen = deepcopy(self.multiple[index])
        chosen.update(**remain)
        self.__dict__.update(**chosen.__dict__)
        self.final = True


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
    source: str = ""

    def __str__(self):
        nentries = len(self.transitions)
        return f"Line list for: {self.name} Formula: {self.formula}, Number of entries: {nentries}"

    def __repr__(self):
        nentries = len(self.transitions)
        return f"Line list name: {self.name}, Number of entries: {nentries}"

    def __len__(self):
        return len(self.transitions)

    def __post_init__(self):
        if len(self.transitions) != 0:
            self.frequencies = [obj.frequency for obj in self.transitions]
            self.catalog_frequencies = [
                obj.catalog_frequency for obj in self.transitions
            ]

    def __eq__(self, other: object) -> bool:
        """
        Dunder method for comparison of LineLists. Since users can accidently
        use different method/formulas yet use the same catalog/lin file to
        create the LineList, we only perform the check on the list of transitions.
        
        Parameters
        ----------
        other : LineList object
            The other LineList to be used for comparison.
            
        Returns
        -------
        bool
            If True, the two LineList objects are equal.
        """
        if not isinstance(other, LineList):
            return NotImplemented
        # Assert that we're comparing LineList objects
        return self.transitions == other.transitions

    def __add__(self, transition_obj: Type[Transition]):
        """
        Dunder method to add Transitions to the LineList.
        
        Parameters
        ----------
        transition_obj : [type]
            [description]
        """
        assert type(transition_obj) == Transition
        self.transitions.append(Transition)

    def __iter__(self):
        """
        Sets up syntax for looping over a LineList. This is recommended more
        for users, but not for programming. When writing new code in the
        module, iterate over the transitions attribute explicitly.
        """
        yield from self.transitions

    @classmethod
    def from_catalog(
        cls,
        name: str,
        formula: str,
        filepath: str,
        min_freq=0.0,
        max_freq=1e12,
        max_lstate=9000.0,
        **kwargs,
    ):
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
        max_lstate: float, optional
            Maximum lower state energy to filter out absurd lines
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
                lambda x: "N'={}, J'={} - N''={}, J''={}".format(*x), axis=1
            )
            # Create a formatted quantum number string
            # catalog_df["qno"] = "N'={}, J'={} - N''={}, J''={}".format(
            #    *catalog_df[["N'", "J'", "N''", "J''"]].values
            # )
            # Calculate E upper to have a complete set of data
            catalog_df["Upper state energy"] = units.calc_E_upper(
                catalog_df["Frequency"], catalog_df["Lower state energy"]
            )
            # Filter out the lower states
            catalog_df = catalog_df.loc[catalog_df["Lower state energy"] <= max_lstate]
            vfunc = np.vectorize(Transition)
            # Vectorized generation of all the Transition objects
            transitions = vfunc(
                catalog_frequency=catalog_df["Frequency"],
                catalog_intensity=catalog_df["Intensity"],
                uncertainty=catalog_df["Uncertainty"],
                lstate_energy=catalog_df["Lower state energy"],
                ustate_energy=catalog_df["Upper state energy"],
                r_qnos=catalog_df["qno"],
                source="Catalog",
                name=name,
                formula=formula,
                uline=False,
                **kwargs,
            )
            linelist_obj = cls(
                name,
                formula,
                filepath=filepath,
                transitions=list(transitions),
                source="Catalog",
            )
            return linelist_obj
        except IndexError:
            return None

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        name="Peaks",
        freq_col="Frequency",
        int_col="Intensity",
        **kwargs,
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
            **kwargs,
        )
        linelist_obj = cls(name=name, transitions=list(transitions), source="Peaks")
        return linelist_obj

    @classmethod
    def from_list(cls, name: str, frequencies: List[float], formula="", **kwargs):
        """
        Generic, low level method for creating a LineList object from a list
        of frequencies. This method can be used when neither lin, catalog, nor
        splatalogue is appropriate and you would like to manually create it
        by handpicked frequencies.

                    obj.uline == True,
            Name of the species - doesn't have to be its real name, just an identifier.
        frequencies: list
            A list of floats corresponding to the "catalog" frequencies.
        formula: str, optional
            Formula of the species, if known.
        kwargs
            Optional settings are passed into the creation of Transition objects.

        Returns
        -------
        LineList
        """
        vfunc = np.vectorize(Transition)
        frequencies = np.asarray(frequencies)
        transitions = vfunc(
            catalog_frequency=frequencies,
            uline=False,
            name=name,
            formula=formula,
            **kwargs,
        )
        linelist_obj = cls(name=name, transitions=list(transitions), source="Catalog")
        return linelist_obj

    @classmethod
    def from_pgopher(cls, name: str, filepath: str, formula="", **kwargs):
        """
        Method to take the output of a PGopher file and create a LineList
        object. The PGopher output must be in the comma delimited specification.
        
        This is actually the ideal way to generate LineList objects: it fills
        in all of the relevant fields, such as linestrength and state energies.
        
        Parameters
        ----------
        name : str
            Name of the molecule
        filepath : str
            Path to the PGopher CSV output
        formula : str, optional
            Chemical formula of the molecule, defaults to an empty string.
            
        Returns
        -------
        LineList
        """
        pgopher_df = pd.read_csv(filepath, skiprows=1)
        pgopher_df = pgopher_df.iloc[:-1]
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            catalog_frequency=pgopher_df["Position"].astype(float),
            catalog_intensity=pgopher_df["Intensity"].astype(float),
            ustate_energy=pgopher_df["Eupper"].apply(units.MHz2cm),
            lstate_energy=pgopher_df["Elower"].apply(units.MHz2cm),
            S=pgopher_df["Spol"],
            uline=False,
            name=name,
            formula=formula,
        )
        linelist_obj = cls(name=name, transitions=list(transitions), source="Catalog")

    @classmethod
    def from_lin(cls, name: str, filepath: str, formula="", **kwargs):
        """
        Generate a LineList object from a .lin file. This method should be used for intermediate assignments, when one
        does not know what the identity of a molecule is but has measured some frequency data.

        Parameters
        ----------
        name : str
            Name of the molecule
        filepath : str
            File path to the .lin file.
        formula : str, optional
            Chemical formula of the molecule if known.
        kwargs
            Additional kwargs are passed into the Transition objects.

        Returns
        -------
        LineList
        """
        lin_df = parsers.parse_lin(filepath)
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            name=name,
            formula=formula,
            catalog_frequency=lin_df["Frequency"],
            uncertainty=lin_df["Uncertainty"],
            r_qnos=lin_df["Quantum numbers"],
            uline=False,
            source="Line file",
            **kwargs,
        )
        linelist_obj = cls(
            name,
            formula,
            filepath=filepath,
            transitions=list(transitions),
            source="Line file",
        )
        return linelist_obj

    @classmethod
    def from_splatalogue_query(cls, dataframe: pd.DataFrame, **kwargs):
        """
        Method for converting a Splatalogue query dataframe into a LineList
        object. This is designed with the intention of pre-querying a set
        of molecules ahead of time, so that the user can have direct control
        over which molecules are specifically targeted without having to
        generate specific catalog files.
        
        Parameters
        ----------
        dataframe : pandas DataFrame
            DataFrame generated by the function `analysis.search_molecule`
        
        Returns
        -------
        LineList
        """
        vfunc = np.vectorize(Transition)
        name = dataframe["Chemical Name"].unique()[0]
        transitions = vfunc(
            name=dataframe["Chemical Name"].values,
            catalog_frequency=dataframe["Frequency"].values,
            catalog_intensity=dataframe["CDMS/JPL Intensity"].values,
            ustate_energy=dataframe["E_U (K)"].values,
            formula=dataframe["Species"].values,
            r_qnos=dataframe["Resolved QNs"].values,
            uline=False,
            source="Splatalogue",
            public=True,
            **kwargs,
        )
        linelist_obj = cls(
            name=name, transitions=list(transitions), source="Splatalogue"
        )
        return linelist_obj

    @classmethod
    def from_artifacts(cls, frequencies: List[float], **kwargs):
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
            public=False,
            **kwargs,
        )
        linelist_obj = cls(
            name="Artifacts", transitions=list(transitions), source="Artifacts"
        )
        return linelist_obj

    @classmethod
    def from_clock(cls, max_multi=64, clock=65000.0, **kwargs):
        """
        Method of generating a LineList object by calculating all possible
        combinations of the
        
        Parameters
        ----------
        max_multi : int, optional
            [description], by default 64
        
        clock : float, optional
            Clock frequency to calculate sub-harmonics of,
            in units of MHz. Defaults to 65,000 MHz, which corresponds
            to the Keysight AWG
        
        Returns
        -------
        LineList object
            LineList object with the full list of possible clock
            spurs, as harmonics, sum, and difference frequencies.
        """
        frequencies = [clock / i for i in range(1, max_multi + 1)]
        for pair in combinations(frequencies, 2):
            # Round to 4 decimal places
            frequencies.append(np.round(sum(pair), 4))
            frequencies.append(np.round(pair[0] - pair[1], 4))
        # Remove duplicates
        frequencies = list(set(frequencies))
        frequencies = sorted(frequencies)
        # Generate Transition objects from this list of frequencies
        vfunc = np.vectorize(Transition)
        transitions = vfunc(
            name="Artifact",
            catalog_frequency=np.asarray(frequencies),
            uline=False,
            source="Artifact",
            public=False,
        )
        linelist_obj = cls(
            name="ClockSpurs", transitions=list(transitions), source="Artifacts"
        )
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

    def to_ftb(self, filepath=None, thres=-10.0, shots=500, dipole=1.0, **kwargs):
        """
        Function to create an FTB file from a LineList object. This will
        create entries for every transition entry above a certain intensity
        threshold, in whatever units the intensities are in; i.e. SPCAT will
        be in log units, while experimental peaks will be in whatever arbitrary
        voltage scale.

        Parameters
        ----------
        filepath: None or str, optional
            Path to write the ftb file to. If None (default), uses the name of
            the LineList and writes to the ftb folder.
        thres: float, optional
            Threshold to cutoff transitions in the ftb file. Transitions with
            less intensity than this value not be included. Units are in the
            same units as whatever the LineList units are.
        shots: int, optional
            Number of shots to integrate.
        dipole: float, optional
            Target dipole moment for the species
        kwargs
            Additional kwargs are passed into the ftb creation, e.g. magnet,
            discharge, etc.
        """
        # If no path is given, use the default naming scheme.
        if filepath is None:
            filepath = f"ftb/{self.name}-batch.ftb"
        # If the source of information are from experimentally measured peaks,
        # we'll use the correct attributes.
        if self.source == "Peaks":
            freq_attr = "frequency"
            int_attr = "intensity"
        else:
            freq_attr = "catalog_frequency"
            int_attr = "catalog_intensity"
        # Get all the frequencies that have intensities above a threshold
        frequencies = [
            getattr(obj, freq_attr)
            for obj in self.transitions
            if getattr(obj, int_attr) >= thres
        ]
        ftb_kwargs = {"dipole": dipole}
        ftb_kwargs.update(**kwargs)
        ftb_str = ""
        # Loop over the frequencies
        for frequency in frequencies:
            ftb_str += fa.generate_ftb_line(
                np.round(frequency, 4), shots=shots, **ftb_kwargs
            )
        with open(filepath, "w+") as write_file:
            write_file.write(ftb_str)

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

    def find_nearest(self, frequency: float, tol=1e-3):
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

    def find_candidates(
        self,
        frequency: float,
        lstate_threshold=4.0,
        freq_tol=1e-1,
        int_tol=-10.0,
        max_uncertainty=0.2,
    ):
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
        # Filter out all transition objects quickly with a list comprehension
        # and if statement
        if self.source == "Peaks":
            freq_attr = "frequency"
        else:
            freq_attr = "catalog_frequency"
        transitions = [
            obj
            for obj in self.transitions
            if all(
                [
                    obj.lstate_energy <= lstate_threshold,
                    obj.catalog_intensity >= int_tol,
                    abs(getattr(obj, freq_attr) - frequency) <= freq_tol,
                    obj.uncertainty <= max_uncertainty,
                ]
            )
        ]
        # If there are candidates, calculate the weights associated with each transition
        if len(transitions) != 0:
            transition_frequencies = np.array(
                [getattr(obj, "freq_attr", np.nan) for obj in transitions]
            )
            transition_intensities = np.array(
                [getattr(obj, "catalog_intensity", np.nan) for obj in transitions]
            )
            # If there are actually no catalog intensities, it should sum up to zero in which case we won't
            # use the intensities in the weight factors
            if np.nansum(transition_intensities) == 0.0:
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

    def update_transition(self, index: int, **kwargs):
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

    def update_linelist(self, transition_objs: List[Transition]):
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

    def get_frequencies(self, numpy=False):
        """
        Method to extract all the frequencies out of a LineList
        
        Parameters
        ----------
        numpy: bool, optional
            If True, returns a NumPy `ndarray` with the frequencies.
        
        Returns
        -------
        List or np.ndarray
            List of transition frequencies
        """
        frequencies = [transition.frequency for transition in self.transitions]
        if numpy:
            frequencies = np.array(frequencies)
        return frequencies

    def get_multiple(self):
        """
        
        Convenience function to extract all the transitions within a LineList
        that have multiple possible assignments.
        
        Returns
        -------
        List
            List of `Transition` objects that have multiple assignments
            remaining.
        """
        assign_objs = list()
        for transition in self.transitions:
            if transition.final == False and len(transition.multiple) != 0:
                assign_objs.append(transition)
        return assign_objs

    def add_uline(self, frequency: float, intensity: float, **kwargs):
        """
        Function to manually add a U-line to the LineList.
        The function creates a Transition object with the frequency and
        intensity values provided by a user, which is then compared with
        the other transition entries within the LineList. If it doesn't
        already exist, it will then add the new Transition to the LineList.
        
        Kwargs are passed to the creation of the Transition object.
        
        Parameters
        ----------
        frequency, intensity: float
            Floats corresponding to the frequency and intensity of the line in
            a given unit.
        """
        transition = Transition(
            frequency=frequency, intensity=intensity, uline=True, **kwargs
        )
        if transition not in self.transitions:
            self.transitions.append(transition)

    def add_ulines(self, data: List[Tuple[float, float]], **kwargs):
        """
        Function to add multiple pairs of frequency/intensity to the current
        LineList.
        
        Kwargs are passed to the creation of the Transition object.
        
        Parameters
        ----------
        data: iterable of 2-tuple
            List-like of 2-tuples corresponding to frequency and intensity.
            Data should look like this example:
            [
                (12345.213, 5.),
                (18623.125, 12.3)
            ]
        """
        for line in data:
            # This assertion makes sure that every line has a specified
            # frequency AND intensity value
            assert len(line) == 2
            self.add_uline(*line, **kwargs)


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
    max_uncertainty : float
        Value to use as the maximum uncertainty for considering a transition
        for assignments.
    """

    experiment: int
    composition: List[str] = field(default_factory=list)
    temperature: float = 4.0
    doppler: float = 0.01
    velocity: float = 0.0
    freq_prox: float = 0.1
    freq_abs: bool = True
    baseline: float = 0.0
    noise_rms: float = 0.0
    noise_region: List[float] = field(default_factory=list)
    max_uncertainty: float = 0.2

    def __str__(self):
        form = (
            f"Experiment: {self.experiment},"
            f"Composition: {self.composition},"
            f"Temperature: {self.temperature} K"
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
    def load_session(cls, filepath: str):
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
        # If the pickle file is just read independently, just make all the
        # folders ahead of time
        folders = [
            "assignment_objs",
            "queries",
            "sessions",
            "clean",
            "figures",
            "reports",
            "logs",
            "outputs",
            "ftb",
            "linelists",
        ]
        for folder in folders:
            if os.path.isdir(folder) is False:
                os.mkdir(folder)
        session._init_logging()
        session.logger.info(f"Reloading session: {filepath}")
        return session

    @classmethod
    def from_ascii(
        cls,
        filepath: str,
        experiment: int,
        composition=["C", "H"],
        delimiter="\t",
        temperature=4.0,
        velocity=0.0,
        col_names=None,
        freq_col="Frequency",
        int_col="Intensity",
        skiprows=0,
        verbose=False,
        **kwargs,
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
        session = cls(
            spec_df,
            experiment,
            composition,
            temperature,
            velocity,
            freq_col,
            int_col,
            verbose,
            **kwargs,
        )
        return session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "log_handlers"):
            for _, handler in self.log_handlers.items():
                handler.close()

    def __repr__(self):
        return f"Experiment {self.session.experiment}"

    def __deepcopy__(self, memodict={}):
        # Kill all of the loggers prior to copying, otherwise thread lock
        # prevents pickling
        if hasattr(self, "log_handlers"):
            for key, handler in self.log_handlers.items():
                handler.close()
        settings = {
            "exp_dataframe": self.data,
            "experiment": self.session.experiment,
            "composition": self.session.composition,
        }
        new_copy = AssignmentSession(**settings)
        new_copy.__dict__.update(**self.__dict__)
        new_copy.session = deepcopy(self.session)
        return new_copy

    def __init__(
        self,
        exp_dataframe: pd.DataFrame,
        experiment: int,
        composition: List[str],
        temperature=4.0,
        velocity=0.0,
        freq_col="Frequency",
        int_col="Intensity",
        verbose=True,
        **kwargs,
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
            "assignment_objs",
            "queries",
            "sessions",
            "clean",
            "figures",
            "reports",
            "logs",
            "outputs",
            "ftb",
            "linelists",
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
        self.t_threshold = self.session.temperature * 3.0
        # Initial threshold for peak detection is set to None
        self.threshold = None
        self.umol_names: Dict[str, str] = dict()
        self.verbose = verbose
        # Holds catalogs
        self.line_lists: Dict[str, Type[LineList]] = dict()
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
        if velocity != 0.0:
            self.set_velocity(velocity)

    def __truediv__(self, other: "AssignmentSession", copy=True):
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
            new_experiment.data[:, self.int_col] = (
                new_experiment[self.int_col] / other.data[other.int_col]
            )
            new_experiment.session.experiment += other.session.experiment
            return new_experiment
        else:
            self.data[:, self.int_col] = (
                self.data[self.int_col] / other.data[other.int_col]
            )

    def __sub__(self, other: "AssignmentSession", copy=True, window=0.2):
        """
        Dunder method to blank the current experiment with another. This method
        will take every detected frequency in the reference experiment, and
        blank the corresponding regions from the current experiment. In effect
        this
        Parameters
        ----------
        other
        copy
        window

        Returns
        -------

        """
        if copy is True:
            experiment = deepcopy(self)
        else:
            experiment = self
        blank_freqs = list()
        # Get regions to blank
        for transition in other.line_lists["Peaks"].transitions:
            # Use the measured frequency where possible
            if transition.frequency != 0.0:
                frequency = transition.frequency
            else:
                frequency = transition.catalog_frequency
            blank_freqs.append(frequency)
        try:
            blank_spectrum = analysis.blank_spectrum(
                self.data,
                blank_freqs,
                self.session.baseline,
                self.session.noise_rms,
                self.freq_col,
                self.int_col,
                window,
                df=False,
            )
            experiment.data[self.int_col] = blank_spectrum
            new_id = self.session.experiment - other.session.experiment
            experiment.session.experiment = new_id
            return experiment
        except AttributeError:
            raise Exception("Peak/Noise detection not yet run!")

    def __contains__(self, item: Union["LineList", str]) -> bool:
        """
        Dunder method to check if a molecule is contained within this experiment.
        
        Parameters
        ----------
        item : Union[LineList, str]
            Item to check; can be a `LineList` or a `str`. If the item is
            a `LineList`, check and see if the `LineList` is present in the
            current list. If the item is a `str`, check in the assignment
            table for a name or a formula.
        
        Returns
        -------
        bool
            True if present in the experiment.
        """
        if isinstance(item, str):
            check = any(item in self.table["name"], item in self.table["formula"])
            return check
        elif isinstance(item, LineList):
            return item in self.line_lists

    def __call__(self, frequency: float, **kwargs) -> pd.DataFrame:
        """
        Dunder method to look up a frequency contained within the
        experiment peak line list.
        
        Additional kwargs are passed into `LineList.find_candidates`
        function, which allows one to specify tolerances for the
        lookup.
        
        Parameters
        ----------
        frequency : float
            Frequency to search the experiment for in MHz.
        
        Returns
        -------
        pd.DataFrame
            DataFrame of the matched frequencies.
        
        Raises
        -------
        AttributeError
            If peak detection has not been run yet
        """
        if "Peaks" in self.line_lists:
            return self.line_lists["Peaks"].find_candidates(frequency, **kwargs)
        else:
            raise AttributeError("Experiment has no peaks!")

    def umol_gen(self, silly=True):
        """
        Generator for unidentified molecule names. Wraps
        Yields
        ------
        str
            Formatted as "UMol_XXX"
        """
        counter = 1
        while counter <= 200:
            # If we want to use silly names from the monsterurl generator
            if silly is True:
                name = get_monster()
            # Otherwise use boring counters
            else:
                name = f"U-molecule-{counter:03d}"
            yield name
            counter += 1

    def create_ulinelist(self, filepath: str, silly=True):
        """
        Create a LineList object for an unidentified molecule. This uses the
        class method `umol_gen` to automatically generate names for U-molecules
        which can then be renamed once it has been identified.

        The session attribute `umol_names` also keeps track of filepaths to
        catalog names. If the filepath has been used previously, then it will
        raise an Exception noting that the filepath is already associated with
        another catalog.

        Parameters
        ----------
        filepath: str
            File path to the catalog or .lin file to use as a reference
        silly: bool, optional
            Flag whether to use boring numbered identifiers, or randomly
            generated `AdjectiveAdjectiveAnimal`.

        Returns
        -------
        LineList object
        """
        if filepath not in self.umol_names:
            path = Path(filepath)
            ext = path.suffix
            name = next(self.umol_gen(silly=silly))
            parameters = {"name": name, "formula": name, "filepath": filepath}
            if ext == ".lin":
                method = LineList.from_lin
            elif ext == ".cat":
                method = LineList.from_catalog
            else:
                raise Exception(f"File extension not recognized: {ext}.")
            linelist = method(**parameters)
            linelist = self.line_lists.get(name, None)
            # Give the first frequency just for book keeping
            frequency = linelist.transitions[0].catalog_frequency
            self.logger.info(
                f"Created a U-molecule: {name}, with frequency {frequency:,.4f}."
            )
            # Create a symlink so we know which catalog this molecule refers to
            sym_path = Path(f"linelists/{name}{ext}")
            sym_path.symlink_to(filepath)
            # Also for internal record keeping
            self.umol_names[filepath] = name
            return linelist
        else:
            name = self.umol_names[filepath]
            raise Exception(f"U-molecule already exists with name: {name}!")

    def rename_umolecule(self, name: str, new_name: str, formula=""):
        """
        Function to update the name of a LineList. This function should be used
        to update a LineList, particularly when the identity of an unidentified
        molecule is discovered.

        Parameters
        ----------
        name: str
            Old name of the LineList.
        new_name: str
            New name of the LineList - preferably, a real molecule name.
        formula: str, optional
            New formula of the LineList.
        """
        if name in self.line_lists:
            # Rename the LineList within the experiment
            self.line_lists[new_name] = self.line_lists.pop(name)
            self.line_lists[new_name].name = new_name
            self.line_lists[new_name].formula = formula
            # Create symlinks to the catalog with the new name for book keeping
            sym_path = Path(f"linelists/{new_name}.cat")
            sym_path.symlink_to(self.line_lists[new_name].filepath)
        else:
            raise Exception(
                f"{name} does not exist in {self.session.experiment} line_lists"
            )

    def add_ulines(self, data: List[Tuple[float, float]], **kwargs):
        """
        Function to manually add multiple pairs of frequency/intensity to the current
        experiment's Peaks list.
        
        Kwargs are passed to the creation of the Transition object.
        
        Parameters
        ----------
        data: iterable of 2-tuple
            List-like of 2-tuples corresponding to frequency and intensity.
            Data should look like this example:
            [
                (12345.213, 5.),
                (18623.125, 12.3)
            ]
        """
        default_dict = {
            "experiment": self.session.experiment,
            "velocity": self.session.velocity,
        }
        default_dict.update(**kwargs)
        if "Peaks" in self.line_lists:
            self.line_lists["Peaks"].add_ulines(data, **default_dict)
        self.logger.info(f"Added {len(data)} transitions to U-lines.")

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
            "stream": logging.INFO,
        }
        logging.captureWarnings(True)
        self.logger = logging.getLogger(f"{self.session.experiment} log")
        self.logger.setLevel(logging.DEBUG)
        # Define file handlers for each type of log
        self.log_handlers = {
            "analysis": logging.FileHandler(
                f"./logs/{self.session.experiment}-analysis.log", mode="w"
            ),
            "warning": logging.FileHandler(
                f"./logs/{self.session.experiment}-warnings.log", mode="w"
            ),
            "debug": logging.FileHandler(
                f"./logs/{self.session.experiment}-debug.log", mode="w"
            ),
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

    def set_velocity(self, value: float):
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
        self.logger.info(f"Set the session velocity to {value}.")

    def detect_noise_floor(self, region=None, als=True, **kwargs):
        """
        Set the noise parameters for the current spectrum. Control over what "defines" the noise floor
        is specified with the parameter region. By default, if region is None then the function will
        perform an initial peak find using 1% of the maximum intensity as the threshold. The noise region
        will be established based on the largest gap between peaks, i.e. hopefully capturing as little
        features in the statistics as possible.
        
        The alternative method is invoked when the `als` argument is set to True, which will use the
        asymmetric least-squares method to determine the baseline. Afterwards, the baseline is decimated
        by an extremely heavy Gaussian blur, and one ends up with a smoothly varying baseline. In this
        case, there is no `noise_rms` attribute to be returned as it is not required to determine the
        minimum peak threshold.

        Parameters
        ----------
        region : 2-tuple or None, optional
            If None, use the automatic algorithm. Otherwise, a 2-tuple specifies the region of the spectrum
            in frequency to use for noise statistics.
        als : bool, optional
            If True, will use the asymmetric least squares method to determine the baseline.
        kwargs
            Additional kwargs are passed into the ALS function.

        Returns
        -------
        baseline - float
            Value of the noise floor
        rms - float
            Noise RMS/standard deviation
        """
        if als is False:
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
                    self.logger.info("Possible noise region:")
                    self.logger.info(freq_values)
                    index = np.argmax(np.diff(freq_values))
                    # Define region as the largest gap
                    region = freq_values[index : index + 2]
                    # Add a 30 MHz offset to either end of the spectrum
                    region[0] = region[0] + 30.0
                    region[1] = region[1] - 30.0
                    # Make sure the frequencies are in ascending order
                    region = np.sort(region)
                    self.logger.info(
                        "Noise region defined as {} to {}.".format(*region)
                    )
                    noise_df = self.data.loc[self.data[self.freq_col].between(*region)]
                    if len(noise_df) < 50:
                        noise_df = self.data.sample(int(len(self.data) * 0.1))
                        self.logger.warning(
                            "Noise region too small; taking a statistical sample."
                        )
                else:
                    # If we haven't found any peaks, sample 10% of random channels and determine the
                    # baseline from those values
                    noise_df = self.data.sample(int(len(self.data) * 0.1))
                    self.logger.warning(
                        "No obvious peaks detected; taking a statistical sample."
                    )
            else:
                # If we haven't found any peaks, sample 10% of random channels and determine the
                # baseline from those values
                noise_df = self.data.sample(int(len(self.data) * 0.1))
                self.logger.warning(
                    "No obvious peaks detected; taking a statistical sample."
                )
                # Calculate statistics
            baseline = np.average(noise_df[self.int_col])
            rms = np.std(noise_df[self.int_col])
            self.session.noise_region = region
            self.session.noise_rms = rms
            self.session.baseline = baseline
            self.logger.info(f"Baseline signal set to {baseline}.")
            self.logger.info(f"Noise RMS set to {rms}.")
        elif als is True:
            # Use the asymmetric least-squares method to determine the baseline
            self.logger.info(
                "Using asymmetric least squares method for baseline determination."
            )
            als_params = {"lam": 1e2, "p": 0.05, "niter": 10}
            als_params.update(**kwargs)
            baseline = fitting.baseline_als(self.data[self.int_col], **als_params)
            # Decimate the noise with a huge Gaussian blur
            baseline = gaussian_filter(baseline, 200.0)
            rms = np.std(baseline)
            self.session.baseline = baseline
            self.session.noise_region = "als"
            self.session.noise_rms = rms
        return baseline, rms

    def find_peaks(
        self, threshold=None, region=None, sigma=6, min_dist=10, als=True, **kwargs
    ):
        """
            Find peaks in the experiment spectrum, with a specified threshold value or automatic threshold.
            The method calls the peak_find function from the analysis module, which in itself wraps peakutils.

            The function works by finding regions of the intensity where the first derivative goes to zero
            and changes sign. This gives peak frequency/intensities from the digitized spectrum, which is
            then "refined" by interpolating over each peak and fitting a Gaussian to determine the peak.

            The peaks are then returned as a pandas DataFrame, which can also be accessed in the peaks_df
            attribute of AssignmentSession.
            
            When a value of threshold is not provided, the function will turn to use automated methods for
            noise detection, either by taking a single value as the baseline (not ALS), or by using the
            asymmetric least-squares method for fitting the baseline. In both instances, the primary intensity
            column to be used for analysis will be changed to "SNR", which is the recommended approach.

            To use the ALS algorithm there may be some tweaking involved for the parameters.
            These are typically found empirically, but for reference here are some "optimal" values
            that have been tested.
            
            For millimeter-wave spectra, larger values of lambda are favored:
            
            lambda = 1e5
            p = 0.1
            
            This should get rid of periodic (fringe) baselines, and leave the "real" signal behind.

            Parameters
            ----------
            threshold : float or None, optional
                Peak detection threshold. If None, will take 1.5 times the noise RMS.
            region : 2-tuple or None, optional
                If None, use the automatic algorithm. Otherwise, a 2-tuple specifies the region of the spectrum
                in frequency to use for noise statistics.
            sigma : float, optional
                Defines the number of sigma (noise RMS) above the baseline to use as the peak detection threshold.
            min_dist : int, optional
                Number of channels between peaks to be detected.
            als : bool, optional
                If True, uses ALS fitting to determine a baseline.
            kwargs
                Additional keyword arguments are passed to the ALS fitting routine.

            Returns
            -------
            peaks_df : dataframe
                Pandas dataframe with Frequency/Intensity columns, corresponding to peaks
        """
        if self.int_col == "SNR":
            # if we run peak find again, the int_col is incorrectly
            # going to be set to SNR giving us bad results. Use the
            # last column instead.
            columns = self.data.columns.to_list()
            columns = [col for col in columns if col != "SNR"]
            self.int_col = columns[-1]
            self.logger.info(f"SNR set as int_col and is invalid for peak finding. Using {self.int_col} instead.")
        if threshold is None and als is False:
            # Use a quasi-intelligent method of determining the noise floor
            # and ultimately using noise + 1 sigma
            baseline, rms = self.detect_noise_floor(region)
            threshold = baseline + (rms * sigma)
            # Convert threshold into SNR units
            threshold /= baseline
            self.data["SNR"] = self.data[self.int_col] / np.abs(baseline)
            self.int_col = "SNR"
            self.logger.info("Now using SNR as primary intensity unit.")
        elif threshold is None and als is True:
            baseline, _ = self.detect_noise_floor(als=True, **kwargs)
            # Convert to SNR, and start using this instead
            self.data["SNR"] = self.data[self.int_col] / np.abs(baseline)
            self.int_col = "SNR"
            self.logger.info("Now using SNR as primary intensity unit.")
            # If using ALS, the sigma argument becomes the threshold value in SNR units
            threshold = sigma
        self.threshold = threshold
        self.logger.info(f"Peak detection threshold is: {threshold}")
        peaks_df = analysis.peak_find(
            self.data,
            freq_col=self.freq_col,
            int_col=self.int_col,
            thres=threshold,
            min_dist=min_dist,
        )
        # Shift the peak intensities down by the noise baseline
        if als is False:
            baseline = getattr(self.session, "baseline", 0.0)
            peaks_df.loc[:, self.int_col] = peaks_df[self.int_col] - baseline
        self.logger.info(f"Found {len(peaks_df)} peaks in total.")
        # Reindex the peaks
        peaks_df.reset_index(drop=True, inplace=True)
        if len(peaks_df) != 0:
            # Generate U-lines
            self.df2ulines(peaks_df, self.freq_col, self.int_col)
            # Assign attribute
            self.peaks = peaks_df
            self.peaks.to_csv(
                f"./outputs/{self.session.experiment}-peaks.csv", index=False
            )
            return peaks_df
        else:
            return None

    def df2ulines(self, dataframe: pd.DataFrame, freq_col=None, int_col=None):
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
            "temperature",
            "doppler",
            "freq_abs",
            "freq_prox",
            "noise_rms",
            "baseline",
            "header",
            "noise_region",
            "composition",
            "name",
            "max_uncertainty",
        ]
        selected_session = {
            key: self.session.__dict__[key]
            for key in self.session.__dict__
            if key not in skip
        }
        # If the Peaks key has not been set up yet, we set it up now
        if "Peaks" not in self.line_lists:
            self.line_lists["Peaks"] = LineList.from_dataframe(
                dataframe,
                name="Peaks",
                freq_col=freq_col,
                int_col=int_col,
                **selected_session,
            )
        # Otherwise, we'll just update the existing LineList
        else:
            vfunc = np.vectorize(Transition)
            transitions = vfunc(
                frequency=dataframe[freq_col],
                intensity=dataframe[int_col],
                **selected_session,
            )
            self.line_lists["Peaks"].update_linelist(transitions)
        new_remaining = len(self.line_lists["Peaks"])
        self.logger.info(f"There are now {new_remaining} line entries in this session.")
        peaks_df = self.line_lists["Peaks"].to_dataframe()[["frequency", "intensity"]]
        peaks_df.columns = ["Frequency", "Intensity"]
        self.peaks = peaks_df

    def search_frequency(self, frequency: float):
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
        slice_df = None
        if self.session.freq_abs is False:
            lower_freq = frequency * (1.0 - self.session.freq_prox)
            upper_freq = frequency * (1 + self.session.freq_prox)
        else:
            lower_freq = frequency - self.session.freq_prox
            upper_freq = frequency + self.session.freq_prox
        if hasattr(self, "table"):
            slice_df = self.table.loc[
                (self.table["frequency"] >= lower_freq)
                & (self.table["frequency"] <= upper_freq)
            ]
        # If no hits turn up, look for it in U-lines
        if slice_df is not None:
            self.logger.info("No assignment found; searching U-lines")
            ulines = np.array(
                [[index, uline.frequency] for index, uline in self.ulines.items()]
            )
            nearest, array_index = routines.find_nearest(ulines[:, 1], frequency)
            uline_index = int(ulines[array_index, 0])
            return nearest, uline_index
        else:
            self.logger.info("Found assignments.")
            return slice_df

    def apply_filter(self, window: Union[str, List[str], np.ndarray], sigma=0.5, int_col=None):
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
        self.logger.info(f"Applying {window} to column {int_col}.")
        # If we have a lis
        if isinstance(window, List):
            for function in window:
                intensity = analysis.filter_spectrum(intensity, function, sigma)
        else:
            intensity = analysis.filter_spectrum(intensity, window, sigma)
        self.data[int_col] = intensity

    def splat_assign_spectrum(self, auto=False):
        """
        Alias for `process_splatalogue`. Function will be removed in a later version.

        Parameters
        ----------
        auto : bool
            Specifies whether the assignment procedure is automatic.
        """
        self.process_splatalogue(auto=auto)

    def process_clock_spurs(self, **kwargs):
        """
        Method that will generate a LineList corresponding to possible
        harmonics, sum, and difference frequencies based on a given clock
        frequency (default: 65,000 MHz).
        
        It is advised to run this function at the end of assignments, owing
        to the sheer number of possible combinations of lines, which may
        interfere with real molecular features.
        
        Parameters
        ----------
        kwargs
            Optional kwargs are passed into the creation of the LineList
            with `LineList.from_clock`.
        """
        self.logger.info(f"Processing electronics RFI peaks.")
        clock_linelist = LineList.from_clock(**kwargs)
        self.process_linelist(name="Clock", linelist=clock_linelist)
        self.logger.info(f"Done processing electronic RFI peaks.")

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
        self.logger.info(f"Beginning Splatalogue lookup on {len(ulines)} lines.")
        iterator = enumerate(ulines)
        if progressbar is True:
            iterator = tqdm(iterator)
        for index, uline in iterator:
            frequency = uline.frequency
            self.logger.info(f"Searching for frequency {frequency:,.4f}")
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
                        for label in [
                            "l-",
                            "c-",
                            "t-",
                            ",",
                            "-gauche",
                            "cis-",
                            "trans-",
                            "trans",
                            "anti",
                            "sym",
                            "=",
                        ]:
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
                            str(atom) in self.session.composition
                            for atom in formula_obj.atoms
                        )
                        if comp_check is False:
                            # If there are crazy things in the mix, forget about it
                            self.logger.info("Molecule " + clean_formula + " rejected.")
                            splat_df.drop(index, inplace=True)
                    except:
                        self.logger.warning(
                            "Could not parse molecule " + clean_formula + " rejected."
                        )
                        splat_df.drop(index, inplace=True)
                nitems = len(splat_df)

                splat_df = analysis.calc_line_weighting(
                    frequency,
                    splat_df,
                    prox=self.session.freq_prox,
                    abs=self.session.freq_abs,
                )
                if splat_df is not None:
                    self.logger.info(
                        f"Found {len(splat_df)} candidates for frequency {frequency:,.4f}, index {index}."
                    )
                    if self.verbose is True:
                        display(HTML(splat_df.to_html()))
                    try:
                        if auto is False:
                            # If not automated, we need a human to look at frequencies
                            # Print the dataframe for notebook viewing
                            splat_index = int(
                                input(
                                    "Please choose an assignment index: 0 - "
                                    + str(nitems - 1)
                                )
                            )
                        else:
                            # If automated, choose closest frequency
                            splat_index = 0
                        self.logger.info(f"Index {splat_index} was chosen.")
                        ass_df = splat_df.iloc[[splat_index]]
                        splat_df.to_csv(
                            f"queries/{self.session.experiment}-{index}.csv",
                            index=False,
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
                            # "S": ass_df["$S_{ij}^2 D^2$"][0],
                            "ustate_energy": ass_df["E_U (K)"][0],
                            "weighting": ass_df["Weighting"][0],
                            "source": "CDMS/JPL",
                            "deviation": frequency - ass_df["Frequency"][0],
                        }
                        # Update the Transition entry
                        uline.__dict__.update(**ass_dict)
                    except ValueError:
                        # If nothing matches, keep in the U-line
                        # pile.
                        self.logger.info(f"Deferring assignment for index {index}.")
                else:
                    # Throw into U-line pile if no matches at all
                    self.logger.info(f"No species known for {frequency:,.4f}")
        self.logger.info("Splatalogue search finished.")

    def overlay_molecule(self, species: str, freq_range=None, threshold=-7.0):
        """
        Function to query splatalogue for a specific molecule. By default, the
        frequency range that will be requested corresponds to the spectral range
        available in the experiment.
        
        Parameters
        ----------
        species : str
            Identifier for a specific molecule, typically name
        
        Returns
        -------
        FigureWidget
            Plotly FigureWidget that shows the experimental spectrum along
            with the detected peaks, and the molecule spectrum.
        DataFrame
            Pandas DataFrame from the Splatalogue query.
        
        Raises
        ------
        Exception
            If no species are found in the query, raises Exception.
        """
        if freq_range is None:
            freq_range = [
                self.data[self.freq_col].min(),
                self.data[self.freq_col].max(),
            ]
        molecule_df = analysis.search_molecule(species, freq_range)
        # Threshold intensity
        molecule_df = molecule_df.loc[molecule_df["CDMS/JPL Intensity"] >= threshold]
        # This prevents missing values from blowing up the normalization
        molecule_df.loc[molecule_df["CDMS/JPL Intensity"] > -1.0][
            "CDMS/JPL Intensity"
        ] = -1.0
        # Calculate the real intensity and normalize
        molecule_df["Intensity"] = 10 ** molecule_df["CDMS/JPL Intensity"]
        molecule_df["Normalized"] = (
            molecule_df["Intensity"] / molecule_df["Intensity"].max()
        )
        # Add annotations to the hovertext
        molecule_df["Annotation"] = (
            molecule_df["Resolved QNs"].astype(str)
            + "-"
            + molecule_df["E_U (K)"].astype(str)
        )
        if len(molecule_df) != 0:
            fig = self.plot_spectrum()
            fig.add_bar(
                x=molecule_df["Frequency"],
                y=molecule_df["Normalized"],
                hoverinfo="text",
                text=molecule_df["Annotation"],
                name=species,
                width=2.0,
            )
            molecule_df.to_csv(f"linelists/{species}-splatalogue.csv", index=False)
            return fig, molecule_df
        else:
            raise Exception(f"No molecules found in Splatalogue named {species}.")

    def process_linelist_batch(self, param_dict=None, yml_path=None, **kwargs):
        """
        Function for processing a whole folder of catalog files. This takes a user-specified
        mapping scheme that will associate catalog files with molecule names, formulas, and
        any other `LineList`/`Transition` attributes. This can be in the form of a dictionary
        or a YAML file; one has to be provided.
        
        An example scheme is given here:
        {
            "cyclopentadiene": {
                "formula": "c5h6",
                "filepath": "../data/catalogs/cyclopentadiene.cat"
            }
        }
        The top dictionary has keys corresponding to the name of the molecule,
        and the value as a sub dictionary containing the formula and filepath
        to the catalog file as minimum input.
        
        You can also provide additional details that are `Transition` attributes:
        {
            "benzene": {
                "formula": "c6h6",
                "filepath": "../data/catalogs/benzene.cat",
                "smiles": "c1ccccc1",
                "publc": False
            }
        }
        
        Parameters
        ----------
        param_dict : dict or None, optional
            If not None, a dictionary containing the mapping scheme will be used
            to process the catalogs. Defaults to None.
        yml_path : str or None, optional
            If not None, corresponds to a str filepath to the YAML file to be read.
        kwargs
            Additional keyword arguments will be passed into the assignment process,
            which are the args for `process_linelist`.
            
        Raises
        ------
        ValueError : If yml_path and param_dict args are the same value.
        """
        if param_dict == yml_path:
            raise ValueError("Please provide arguments to param_dict or yml_path.")
        if yml_path:
            param_dict = routines.read_yaml(yml_path)
            yml_path = Path(yml_path)
            root = yml_path.parents[0]
        linelists = list()
        self.logger.info(f"Processing {len(param_dict)} catalog entries in batch mode.")
        for name, subdict in param_dict.items():
            try:
                # Append the directory path to the catalog filepath
                subdict["filepath"] = str(root.joinpath(subdict["filepath"]))
                self.logger.info(f"Creating LineList for molecule {name}")
                extension = Path(subdict["filepath"]).suffix
                if extension == ".cat":
                    func = LineList.from_catalog
                elif extension == ".lin":
                    func = LineList.from_lin
                else:
                    raise ValueError(
                        f"File extension not recognized: {name} extension: {extension}"
                    )
                linelist_obj = func(name=name, **subdict)
                print(linelist_obj)
                self.process_linelist(name=linelist_obj.name, linelist=linelist_obj)
            except ValueError:
                self.logger.warning(f"Could not parse molecule {name}.")
                raise Exception(f"Could not parse molecule {name} from dictionary.")
        self.logger.info("Completed catalog batch analysis.")

    def process_linelist(
        self,
        name=None,
        formula=None,
        filepath=None,
        linelist=None,
        auto=True,
        thres=-10.0,
        progressbar=True,
        tol=None,
        **kwargs,
    ):
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
        self.logger.info(f"Processing local catalog for molecule {name}.")
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
                raise Exception(
                    "File extension for reference line list not recognized!"
                )
            linelist = func(
                name=name,
                formula=formula,
                filepath=filepath,
                min_freq=self.data[self.freq_col].min(),
                max_freq=self.data[self.freq_col].max(),
            )
            if name not in self.line_lists:
                self.line_lists[name] = linelist
        else:
            raise Exception("Please specify an internal or external line list!")
        if linelist is not None:
            nassigned = 0
            # Sort the LineList by intensity if possible. If the strongest line isn't
            # there, the other lines shouldn't be there
            if linelist.source in ["Splatalogue", "Catalog"]:
                linelist.transitions = sorted(
                    linelist, key=lambda line: line.catalog_intensity
                )[::-1]
            # Filter out out-of-band stuff, but with some small tolerance extending
            # out just in case it's close
            max_freq = max(self.line_lists["Peaks"].get_frequencies()) * 1.001
            min_freq = min(self.line_lists["Peaks"].get_frequencies()) * 0.999
            # Make sure we're only assigning with in-band, has a suitable uncertainty,
            # and that the lower state energy is within range
            transitions = [
                transition
                for transition in linelist.transitions
                if all(
                    [
                        min_freq <= transition.catalog_frequency <= max_freq,
                        transition.uncertainty <= self.session.max_uncertainty,
                        transition.lstate_energy <= self.t_threshold,
                    ]
                )
            ]
            # Loop over the LineList lines
            if progressbar is True:
                iterator = tqdm(transitions)
                iterator = enumerate(iterator)
            else:
                iterator = enumerate(transitions)
            # Loop over all of the U-lines
            for index, transition in iterator:
                # Control the flow so that we're not wasting time looking for lines if the strongest
                # transitions are not seen
                # if (index > 5) and (nassigned == 0) and (linelist.source in ["Catalog", "Splatalogue"]):
                #    self.logger.info(
                #    "Searched for five strongest transitions in {linelist.name}, and nothing; aborting."
                #    )
                #    break
                # If no value of tolerance is provided, determine from the session
                if tol is None:
                    if self.session.freq_abs is True:
                        tol = self.session.freq_prox
                    else:
                        tol = (1.0 - self.session.freq_prox) * transition.frequency
                # Log the search
                self.logger.info(
                    f"Searching for frequency {transition.catalog_frequency:,.4f}"
                    f" with tolerances: {self.t_threshold:.2f} K, "
                    f" +/-{tol:.4f} MHz, {thres} intensity."
                )
                # Find transitions in the peak list that might match
                can_pkg = self.line_lists["Peaks"].find_candidates(
                    transition.catalog_frequency,
                    lstate_threshold=self.t_threshold,
                    freq_tol=tol,
                    int_tol=thres,
                )
                # If there are actual candidates instead of NoneType, we can process it.
                if can_pkg is not None:
                    candidates, weighting = can_pkg
                    ncandidates = len(candidates)
                    self.logger.info(f"Found {ncandidates} possible matches.")
                    make_assignment = True
                    # If auto mode or if there's just one candidate, just take the highest weighting
                    if auto is True:
                        chosen = candidates[weighting.argmax()]
                        if chosen.uline is False:
                            # If this line has previously been assigned, then we
                            # fill up the multiple "buffer"
                            if not chosen.check_molecule(transition):
                                if transition not in chosen.multiple:
                                    # Only add more transitions if this is a
                                    # different molecule and it's not already in
                                    # the list
                                    chosen.multiple.append(transition)
                            make_assignment = False
                    elif auto is False:
                        for cand_idx, candidate in enumerate(candidates):
                            print(cand_idx, candidate)
                        chosen_idx = int(
                            input("Please specify the candidate index.   ")
                        )
                        chosen = candidates[chosen_idx]
                    if make_assignment is True:
                        self.logger.info(
                            f"Assigning {transition.name}"
                            f" (catalog {transition.catalog_frequency:,.4f})"
                            f" to peak {index} at {chosen.frequency:,.4f}."
                        )
                        # Create a copy of the Transition data from the LineList
                        assign_dict = copy(transition.__dict__)
                        # Update with the measured frequency and intensity
                        assign_dict["frequency"] = chosen.frequency
                        assign_dict["intensity"] = chosen.intensity
                        assign_dict["experiment"] = chosen.experiment
                        assign_dict["velocity"] = self.session.velocity
                        assign_dict["peak_id"] = chosen.peak_id
                        assign_dict["uline"] = False
                        # Add any other additional kwargs
                        assign_dict.update(**kwargs)
                        # Copy over the information from the assignment, and update
                        # the experimental peak information with the assignment
                        chosen.__dict__.update(**assign_dict)
                        nassigned += 1
            self.logger.info(
                f"Assigned {nassigned} new transitions to {linelist.name}."
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
        # Get the U-line list
        ulines = self.line_lists.get("Peaks", None).get_ulines()
        old_nulines = len(ulines.get_ulines())
        db = database.SpectralCatalog(dbpath)
        self.logger.info(f"Processing local database: {dbpath}")
        for index, uline in tqdm(enumerate(ulines)):
            self.logger.info(f"Searching database for {uline.frequency:.4f}")
            catalog_df = db.search_frequency(
                uline.frequency, self.session.freq_prox, self.session.freq_abs
            )
            if catalog_df is not None:
                catalog_df["frequency"].replace(0.0, np.nan, inplace=True)
                catalog_df["frequency"].fillna(
                    catalog_df["catalog_frequency"], inplace=True
                )
                if len(catalog_df) > 0:
                    sliced_catalog = analysis.line_weighting(
                        uline.frequency,
                        catalog_df,
                        prox=self.session.freq_prox,
                        abs=self.session.freq_abs,
                        freq_col="frequency",
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
                            "index": index,
                            "frequency": uline.frequency,
                            "source": "Database",
                            "deviation": uline.frequency - select_df["frequency"],
                            "uline": False,
                        }
                        assign_dict = select_df.to_dict()
                        assign_dict.update(**new_dict)
                        # Use assign_line function to mark peak as assigned
                        uline.__dict__.update(**assign_dict)
                    else:
                        raise IndexError("Invalid index chosen for assignment.")
        # Update the internal table
        ass_df = pd.DataFrame(data=[ass_obj.__dict__ for ass_obj in self.assignments])
        remaining_ulines = self.line_lists.get("Peaks").get_ulines()
        self.table = ass_df
        self.logger.info(f"Prior number of ulines: {old_nulines}")
        self.logger.info(f"Current number of ulines: {len(remaining_ulines)}")
        self.logger.info("Finished processing local database.")

    def copy_assignments(self, other: "AssignmentSession", thres_prox=1e-2):
        """
        Function to copy assignments from another experiment. This class
        method wraps two analysis routines: first, correlations in detected
        peaks are found, and indexes of where correlations are found will be
        used to locate the corresponding Transition object, and copy its data
        over to the current experiment.

        Parameters
        ----------
        other : AssignmentSession object
            The reference AssignmentSession object to copy assignments from
        thres_prox : float, optional
            Threshold for considering coincidences between spectra.
        """
        self.logger.info(
            f"Copying assignments from Experiment {other.session.experiment}"
        )
        _, indices = analysis.correlate_experiments(
            [self, other], thres_prox=thres_prox
        )
        # Get the correlation matrix
        corr_mat = indices[other.session.experiment][1]
        analysis.copy_assignments(self, other, corr_mat)
        n_assign = len(np.where(corr_mat > 0)[0])
        self.logger.info(f"Copied {n_assign} assignments.")

    def blank_spectrum(self, noise=0.0, noise_std=0.05, window=1.0):
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
        """
        sources = ["CDMS/JPL", "Literature", "New"]
        slices = [
            self.table.loc[self.table["source"] == "CDMS/JPL"],
            self.table.loc[
                (self.table["source"] != "CDMS/JPL") & (self.table["public"] == True)
            ],
            self.table.loc[
                (self.table["source"] != "CDMS/JPL") & (self.table["public"] == False)
            ],
        ]
        last_source = "Catalog"
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
                        df=False,
                    )
                    self.data[source] = blanked_spectrum
                    last_source = source
            except (KeyError, ValueError):
                self.logger.warning(
                    f"Could not blank spectrum {last_source} with {source}."
                )

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
        self.identifications = {name: names.count(name) for name in self.names}
        return self.identifications

    def create_uline_ftb_batch(
        self,
        filepath=None,
        shots=500,
        dipole=1.0,
        threshold=0.0,
        sort_int=False,
        atten=None,
    ):
        """
        Create an FTB file for use in QtFTM based on the remaining ulines. This is used to provide cavity
        frequencies.
        
        If a filepath is not specified, a -uline.ftb file will be created in the
        ftb folder.
        
        The user has the ability to control parameters of the batch by setting
        a global shot count, dipole moment, and minimum intensity value for
        creation.

        Parameters
        ----------
        filepath : str or None, optional
            Path to save the .ftb file to. If None, defaults to the session ID.
        shots : int, optional
            Number of shots to integrate on each frequency
        dipole : float, optional
            Dipole moment in Debye attenuation target for each frequency
        threshold : float, optional
            Minimum value for the line intensity to be considered. For example,
            if the spectrum is analyzed in units of SNR, this would be the minimum
            value of SNR to consider in the FTB file.
        sort_int : bool, optional
            If True, sorts the FTB entries in descending intensity order.
        atten : None or int, optional
            Value to use for the attenuation, overwriting the `dipole` argument. This is
            useful for forcing cavity power in the high band.
        """
        if filepath is None:
            filepath = f"./ftb/{self.session.experiment}-ulines.ftb"
        lines = ""
        transitions = self.line_lists["Peaks"].get_ulines()
        transitions = [
            transition
            for transition in transitions
            if transition.intensity >= threshold
        ]
        if atten is not None:
            assert type(atten) == int
            ftb_settings = {"atten": atten}
        else:
            ftb_settings = {"dipole": dipole}
        # Sort the intensities such that the strongest ones are scanned first.
        if sort_int is True:
            transitions = sorted(transitions, key=lambda line: line.intensity)[::-1]
        for index, uline in enumerate(transitions):
            if uline.intensity >= threshold:
                lines += fa.generate_ftb_line(uline.frequency, shots, **ftb_settings)
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def create_uline_dr_batch(
        self,
        filepath=None,
        select=None,
        shots=25,
        dipole=1.0,
        min_dist=500.0,
        thres=None,
        atten=None,
        drpower=13,
    ):
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
        atten: None or int, optional
            Value to use for the attenuation, overwriting the `dipole` argument. This is
            useful for forcing cavity power in the high band.
        """
        ulines = self.line_lists["Peaks"].get_ulines()
        if select is None:
            cavity_freqs = [uline.frequency for uline in ulines]
            dr_freqs = [uline.frequency for uline in ulines]
        else:
            cavity_freqs = select
            dr_freqs = select
        if thres is not None:
            intensities = np.array([uline.intensity for uline in ulines])
            mask = np.where(intensities >= thres)
            dr_freqs = np.asarray(dr_freqs)[mask]
            cavity_freqs = np.asarray(cavity_freqs)
        ftb_settings = {"drpower": -20, "skiptune": False, "dipole": dipole}
        if atten is not None:
            assert type(atten) == int
            del ftb_settings["dipole"]
            ftb_settings["atten"] = int(atten)
        lines = ""
        for cindex, cavity in enumerate(cavity_freqs):
            for dindex, dr in enumerate(dr_freqs):
                if dindex == 0:
                    ftb_settings.update(**{"drpower": -20, "skiptune": False})
                    lines += fa.generate_ftb_line(cavity, shots, **ftb_settings)
                if np.abs(cavity - dr) >= min_dist:
                    ftb_settings.update(
                        **{"drpower": drpower, "skiptune": True, "drfreq": dr}
                    )
                    lines += fa.generate_ftb_line(cavity, shots, **ftb_settings)
        if filepath is None:
            filepath = f"ftb/{self.session.experiment}-dr.ftb"
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def create_full_dr_batch(
        self,
        cavity_freqs: List[float],
        filepath=None,
        shots=25,
        dipole=1.0,
        min_dist=500.0,
        atten=None,
        drpower=13,
    ):
        """
        Create an FTB batch file for use in QtFTM to perform a DR experiment.
        A list of selected frequencies can be used as the cavity frequencies, which will
        subsequently be exhaustively DR'd against by ALL frequencies in the
        experiment.

        The file is then saved to "ftb/XXX-full-dr.ftb".
        
        The ``atten`` parameter provides a more direct way to control RF power;
        if this value is used, it will overwrite the dipole moment setting.

        Parameters
        ----------
        cavity_freqs : iterable of floats
            Iterable of frequencies to tune to, in MHz.
        filepath : str, optional
            Path to save the ftb file to. Defaults to ftb/{}-dr.ftb
        shots : int, optional
            Number of integration shots
        dipole : float, optional
            Dipole moment used for attenuation setting
        min_dist : float, optional
            Minimum frequency difference between cavity and DR frequency to actually perform
            the experiment
        atten : None or int, optional
            Value to set the rf attenuation. By default, this is None, which will use the
            dipole moment instead to set the rf power. If a value is provided, it will
            overwrite whatever the dipole moment setting is.
        """
        dr_freqs = np.array(self.line_lists["Peaks"].get_frequencies())
        lines = ""
        ftb_settings = {"drpower": -20, "skiptune": False, "dipole": dipole}
        if atten is not None:
            assert type(atten) == int
            del ftb_settings["dipole"]
            ftb_settings["atten"] = int(atten)
        for cindex, cavity in enumerate(cavity_freqs):
            for dindex, dr in enumerate(dr_freqs):
                if dindex == 0:
                    ftb_settings.update(**{"drpower": -20, "skiptune": False})
                    lines += fa.generate_ftb_line(cavity, shots, **ftb_settings)
                if np.abs(cavity - dr) >= min_dist:
                    ftb_settings.update(
                        **{"drpower": drpower, "skiptune": True, "drfreq": dr}
                    )
                    lines += fa.generate_ftb_line(cavity, shots, **ftb_settings)
        if filepath is None:
            filepath = f"ftb/{self.session.experiment}-full-dr.ftb"
        with open(filepath, "w+") as write_file:
            write_file.write(lines)

    def analyze_molecule(
        self, Q=None, T=None, name=None, formula=None, smiles=None, chi_thres=10.0
    ):
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
            raise Exception(
                "No valid selector specified! Please give a name, formula, or SMILES code."
            )
        # Loop over all of the assignments
        mol_data = list()
        for index, ass_obj in enumerate(self.assignments):
            # If the assignment matches the criteria
            # we perform the analysis
            if ass_obj.__dict__[selector] == value:
                self.logger.info(
                    f"Performing line profile analysis on assignment index {index}."
                )
                # Perform a Gaussian fit whilst supplying as much information as we can
                # The width is excluded because it changes significantly between line profiles
                fit_result, summary = analysis.fit_line_profile(
                    self.data,
                    center=ass_obj.frequency,
                    intensity=ass_obj.intensity,
                    freq_col=self.freq_col,
                    int_col=self.int_col,
                    logger=self.logger,
                )
                # If the fit actually converged and worked
                if fit_result:
                    # Calculate what the lab frame frequency would be in order to calculate the frequency offset
                    lab_freq = fit_result.best_values["center"] + units.dop2freq(
                        self.session.velocity, fit_result.best_values["center"]
                    )
                    summary["Frequency offset"] = lab_freq - ass_obj.catalog_frequency
                    summary["Doppler velocity"] = units.freq2vel(
                        ass_obj.catalog_frequency, summary["Frequency offset"]
                    )
                    if Q is not None and T is not None:
                        # Add the profile parameters to list
                        profile_dict = aa.lineprofile_analysis(
                            fit_result, ass_obj.I, Q, T, ass_obj.ustate_energy
                        )
                        ass_obj.N = profile_dict["N cm$^{-2}$"]
                        summary.update(profile_dict)
                    ass_obj.fit = fit_result
                    mol_data.append(summary)
        # If there are successful analyses performed, format the results
        if len(mol_data) > 0:
            profile_df = pd.DataFrame(data=mol_data)
            # Sort the dataframe by ascending order of chi square - better fits are at the top
            profile_df.sort_values(["Chi squared"], inplace=True)
            # Threshold the dataframe to ensure good statistics
            profile_df = profile_df.loc[profile_df["Chi squared"] <= chi_thres]
            # Calculate the weighted average VLSR based on the goodness-of-fit
            profile_df.loc[:, "Weight"] = (
                profile_df["Chi squared"].max() / profile_df["Chi squared"].values
            )
            weighted_avg = np.sum(
                profile_df["Weight"] * profile_df["Doppler velocity"]
            ) / np.sum(profile_df["Weight"])
            # Calculate the weighted standard deviation
            stdev = np.sum(
                profile_df["Weight"]
                * (profile_df["Doppler velocity"] - weighted_avg) ** 2
            ) / np.sum(profile_df["Weight"])
            self.logger.info(
                f"Calculated VLSR: {weighted_avg:.3f}+/-{stdev:.3f}"
                f" based on {len(profile_df)} samples."
            )
            return_data = [profile_df, ufloat(weighted_avg, stdev)]
            # If there's data to perform a rotational temperature analysis, then do it!
            if "L" in profile_df.columns:
                self.logger.info("Performing rotational temperature analysis.")
                rot_temp = aa.rotational_temperature_analysis(
                    profile_df["L"], profile_df["E upper"]
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
                if len(obj.multiple) != 0 and obj.final is False:
                    warnings.warn(
                        f"Transition at {obj.frequency:4f} has multiple candidates."
                    )
                    warnings.warn(f"Please choose assignment for peak {obj.peak_id}.")
                else:
                    # Set the transition as finalized
                    obj.final = True
                # Dump all the assignments into YAML format
                obj.to_file(f"assignment_objs/{obj.experiment}-{obj.peak_id}", "yaml")
                obj.deviation = obj.catalog_frequency - obj.frequency
            # Convert all of the assignment data into a CSV file
            assignment_df = pd.DataFrame(data=[obj.__dict__ for obj in assignments])
            self.table = assignment_df
            self.table.sort_values(["frequency"], ascending=True, inplace=True)
            # Generate a LaTeX table for publication
            self.create_latex_table()
            # Dump assignments to disk
            assignment_df.to_csv(f"reports/{self.session.experiment}.csv", index=False)
            # Dump Uline data to disk
            peak_data = [[peak.frequency, peak.intensity] for peak in ulines]
            peak_df = pd.DataFrame(peak_data, columns=["Frequency", "Intensity"])
            peak_df.to_csv(f"reports/{self.session.experiment}-ulines.csv", index=False)

            tally = self._get_assigned_names()
            combined_dict = {
                "assigned_lines": len(assignments),
                "ulines": len(ulines),
                "peaks": self.line_lists["Peaks"].get_frequencies(),
                "num_peaks": len(self.line_lists["Peaks"]),
                "tally": tally,
                "unique_molecules": self.names,
                "num_unique": len(self.names),
            }
            # Combine Session information
            combined_dict.update(self.session.__dict__)
            # Dump to disk
            routines.dump_yaml(f"sessions/{self.session.experiment}.yml", "yaml")
            self._create_html_report()
            # Dump data to notebook output
            for key, value in combined_dict.items():
                self.logger.info(key + ":   " + str(value))
        else:
            raise Exception(
                "No assignments made in this session - nothing to finalize!"
            )

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

    def simulate_sticks(
        self,
        catalogpath: str,
        N: float,
        Q: float,
        T: float,
        doppler=None,
        gaussian=False,
    ):
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
            (catalog_df["Frequency"] >= self.data[self.freq_col].min())
            & (catalog_df["Frequency"] <= self.data[self.freq_col].max())
        ]
        if gaussian is False:
            return catalog_df[["Frequency", "Flux (Jy)"]]
        else:
            # Convert Doppler width to frequency widths
            widths = units.dop2freq(doppler, catalog_df["Frequency"].values)
            # Calculate the Gaussian amplitude
            amplitudes = catalog_df["Flux (Jy)"] / np.sqrt(2.0 * np.pi ** 2.0 * widths)
            sim_y = self.simulate_spectrum(
                self.data[self.freq_col],
                catalog_df["Frequency"].values,
                widths,
                amplitudes,
            )
            simulated_df = pd.DataFrame(
                data=list(zip(self.data[self.freq_col], sim_y)),
                columns=["Frequency", "Flux (Jy)"],
            )
            return simulated_df

    def simulate_spectrum(
        self,
        x: np.ndarray,
        centers: List[float],
        widths: List[float],
        amplitudes: List[float],
        fake=False,
    ):
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
                scaling = 1.0
            y += scaling * model.eval(x=x, center=c, sigma=w, amplitude=a)
        return y

    def clean_spectral_assignments(self, window=1.0):
        """
        Function to blank regions of the spectrum that have already been
        assigned. This function takes the frequencies of assignments, and
        uses the noise statistics to generate white noise to replace the
        peak. This is to let one focus on unidentified features, rather than
        be distracted by the assignments with large amplitudes.

        Parameters
        ----------
        window: float, optional
            Frequency value in MHz to blank. The region corresponds to the
            frequency +/- this value.
        """
        # Make a back up of the full spectrum
        if "Full" not in self.data.columns:
            self.data["Full"] = self.data[self.int_col].copy()
        # If the backup exists, restore the backup first before blanking.
        else:
            self.data[self.int_col] = self.data["Full"].copy()
        assignments = self.line_lists["Peaks"].get_assignments()
        frequencies = np.array([transition.frequency for transition in assignments])
        self.logger.info(f"Blanking spectrum over {frequencies} windows.")
        if self.session.noise_region == "als":
            baseline = np.mean(self.session.baseline)
            baseline /= baseline
            rms = np.std(baseline)
        else:
            baseline = self.session.baseline
            rms = self.session.noise_rms
        self.data[self.int_col] = analysis.blank_spectrum(
            self.data,
            frequencies,
            baseline,
            rms,
            freq_col=self.freq_col,
            int_col=self.int_col,
            window=window,
            df=False,
        )

    def calculate_assignment_statistics(self):
        """
        Function for calculating some aggregate statistics of the assignments and u-lines. This
        breaks the assignments sources up to identify what the dominant source of information was.
        The two metrics for assignments are the number of transitions and the intensity contribution
        assigned by a particular source.
        :return: dict
        """
        reduced_table = self.table[
            [
                "frequency",
                "intensity",
                "formula",
                "name",
                "catalog_frequency",
                "deviation",
                "ustate_energy",
                "source",
                "public",
            ]
        ]
        artifacts = reduced_table.loc[reduced_table["name"] == "Artifact"]
        splat = reduced_table.loc[reduced_table["source"] == "CDMS/JPL"]
        local = reduced_table.loc[
            (reduced_table["source"] != "Artifact")
            & (reduced_table["source"] != "CDMS/JPL")
        ]
        public = local.loc[local["public"] == True]
        private = local.loc[local["public"] == False]
        sources = [
            "Artifacts",
            "Splatalogue",
            "Published molecules",
            "Unpublished molecules",
        ]
        # Added up the total number of lines
        total_lines = len(self.line_lists["Peaks"])
        # Add up the total intensity
        ulines = self.line_lists.get("Peaks").get_ulines()
        total_intensity = np.sum([uline.intensity for uline in ulines])
        total_intensity += np.sum(reduced_table["intensity"])
        line_breakdown = [len(source) for source in [artifacts, splat, public, private]]
        intensity_breakdown = [
            np.sum(source["intensity"])
            for source in [artifacts, splat, public, private]
        ]
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
                "cumulative intensity breakdown": cum_int_breakdown,
            },
            # These are the corresponding values in percentage
            "percent": {
                "line breakdown": [
                    (value / total_lines) * 100.0 for value in line_breakdown
                ],
                "intensity breakdown": [
                    (value / total_intensity) * 100.0 for value in intensity_breakdown
                ],
                "cumulative line breakdown": [
                    (value / total_lines) * 100.0 for value in cum_line_breakdown
                ],
                "cumulative intensity breakdown": [
                    (value / total_intensity) * 100.0 for value in cum_int_breakdown
                ],
            },
            "molecules": {
                "CDMS/JPL": {
                    name: len(splat.loc[splat["name"] == name])
                    for name in splat["name"].unique()
                },
                "Published": {
                    name: len(public.loc[public["name"] == name])
                    for name in public["name"].unique()
                },
                "Unpublished": {
                    name: len(private.loc[private["name"] == name])
                    for name in private["name"].unique()
                },
            },
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
        fig.layout["yaxis"]["title"] = str(self.int_col)

        fig.add_scattergl(
            x=self.data[self.freq_col],
            y=self.data[self.int_col],
            name="Experiment",
            opacity=0.6,
        )

        if "Peaks" in self.line_lists:
            ulines = self.line_lists["Peaks"].get_ulines()
            labels = list(range(len(ulines)))
            amplitudes = np.array([uline.intensity for uline in ulines])
            centers = np.array([uline.frequency for uline in ulines])
            # Add sticks for U-lines
            fig.add_bar(
                x=centers, y=amplitudes, hoverinfo="text", text=labels, name="Peaks"
            )

            if simulate is True:
                widths = units.dop2freq(self.session.doppler, centers)

                simulated = self.simulate_spectrum(
                    self.data[self.freq_col].values,
                    centers,
                    widths,
                    amplitudes,
                    fake=True,
                )

                self.simulated = pd.DataFrame(
                    data=list(zip(self.data[self.freq_col].values, simulated)),
                    columns=["Frequency", "Intensity"],
                )

                fig.add_scattergl(
                    x=self.simulated["Frequency"],
                    y=self.simulated["Intensity"],
                    name="Simulated spectrum",
                )

        return fig

    def _create_html_report(self, filepath=None):
        """
        Function for generating an HTML report for sharing. The HTML report is rendered with
        Jinja2, and uses the template "report_template.html" located in the module directory.
        The report includes interactive plots showing statistics of the assignments/ulines and
        an overview of the spectrum. At the end of the report is a table of the assignments and
        uline data.
        
        Parameters
        ----------
        
        filepath: str or None, optional
            Path to save the report to. If `None`, defaults to `reports/{id}-summary.html`
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
            [
                "frequency",
                "intensity",
                "formula",
                "name",
                "catalog_frequency",
                "deviation",
                "ustate_energy",
                "source",
            ]
        ]
        # Render pandas dataframe HTML with bar annotations
        reduced_table_html = (
            reduced_table.style.bar(
                subset=["deviation", "ustate_energy"],
                align="mid",
                color=["#d65f5f", "#5fba7d"],
            )
            .bar(subset=["intensity"], color="#5fba7d")
            .format(
                {
                    "frequency": "{:.4f}",
                    "catalog_frequency": "{:.4f}",
                    "deviation": "{:.3f}",
                    "ustate_energy": "{:.2f}",
                    "intensity": "{:.3f}",
                }
            )
            .set_table_attributes("""class = "data-table hover compact" """)
            .render(classes=""" "data-table hover compact" """)
        )
        html_dict["assignments_table"] = reduced_table_html
        # The unidentified features table
        ulines = self.line_lists["Peaks"].get_ulines()
        uline_df = pd.DataFrame(
            [[uline.frequency, uline.intensity] for uline in ulines],
            columns=["Frequency", "Intensity"],
        )
        html_dict["uline_table"] = (
            uline_df.style.bar(subset=["Intensity"], color="#5fba7d")
            .format({"Frequency": "{:.4f}", "Intensity": "{:.2f}"})
            .set_table_attributes("""class = "data-table hover compact" """)
            .render(classes=""" "data-table hover compact" """)
        )
        # Plotly displays of the spectral feature breakdown and whatnot
        html_dict["plotly_breakdown"] = plot(self.plot_breakdown(), output_type="div")
        html_dict["plotly_figure"] = plot(self.plot_assigned(), output_type="div")
        # Render the template with Jinja and save the HTML report
        output = template.render(session=self.session, **html_dict)
        if filepath is None:
            filepath = f"reports/{self.session.experiment}-summary.html"
        with open(filepath, "w+") as write_file:
            write_file.write(output)

    def create_latex_table(self, filepath=None, header=None, cols=None, **kwargs):
        """
        Method to create a LaTeX table summarizing the measurements in this experiment.
        
        Without any additional inputs, the table will be printed into a .tex file
        in the reports folder. The table will be created with the minimum amount
        of information required for a paper, including the frequency and intensity
        information, assignments, and the source of the information.
        
        The user can override the default settings by supplying `header` and `col`
        arguments, and any other kwargs are passed into the `to_latex` pandas
        DataFrame method. The header and col lengths must match.
        
        Parameters
        ----------
        filepath : str, optional
            Filepath to save the LaTeX table to; by default None
        header : iterable of str, optional
            An iterable of strings specifying the header to be printed. By default None
        cols : iterable of str, optional
            An iterable of strings specifying which columns to include. If this is
            changed, the header must also be changed to reflect the new columns.
        """
        table = self.line_lists["Peaks"].to_dataframe()
        # Numerical formatting
        formatters = {
            "frequency": "{:,.4f}".format,
            "intensity": "{:.2f}".format,
            "catalog_frequency": "{:,.4f}".format,
            "deviation": "{:,.4f}".format,
        }

        def ref_formatter(x):
            noform = ["CDMS/JPL", "Artifact", "U"]
            if x not in noform:
                return f"\cite{x}"
            else:
                return x

        # Replace the source information to designate u-line if it is a u-line
        table.loc[table["uline"] == True, "source"] = "U"
        if header is None:
            header = [
                "Frequency",
                "Intensity",
                "Catalog frequency",
                "Deviation",
                "Name",
                "Formula",
                "Quantum numbers",
                "Source",
            ]
        if cols is None:
            cols = [
                "frequency",
                "intensity",
                "catalog_frequency",
                "deviation",
                "name",
                "formula",
                "r_qnos",
                "source",
            ]
        assert len(header) == len(cols)
        if "source" in cols:
            table["source"].apply(ref_formatter)
        if "formula" in cols:
            table["formula"].apply(lambda x: f"\ce{x}")
        if filepath is None:
            filepath = f"reports/{self.session.experiment}-table.tex"
        # Settings to be passed into the latex table creation
        table_settings = {
            "formatters": formatters,
            "index": False,
            "longtable": True,
            "header": header,
            "escape": False,
        }
        table_settings.update(**kwargs)
        # Write the table to file
        with open(filepath, "w+") as write_file:
            table[cols].to_latex(buf=write_file, **table_settings)

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
            nrows=1,
            ncols=2,
            **{"subplot_titles": ("Reference Breakdown", "Intensity Breakdown")},
        )
        fig.layout["title"] = "Spectral Feature Breakdown"
        fig.layout["showlegend"] = False
        reduced_table = self.table[
            [
                "frequency",
                "intensity",
                "formula",
                "name",
                "catalog_frequency",
                "deviation",
                "ustate_energy",
                "source",
                "public",
            ]
        ]
        artifacts = reduced_table.loc[reduced_table["name"] == "Artifact"]
        splat = reduced_table.loc[reduced_table["source"] == "CDMS/JPL"]
        local = reduced_table.loc[
            (reduced_table["source"] != "Artifact")
            & (reduced_table["source"] != "CDMS/JPL")
        ]
        public = local.loc[local["public"] == True]
        private = local.loc[local["public"] == False]
        sources = [
            "Artifacts",
            "Splatalogue",
            "Published molecules",
            "Unpublished molecules",
        ]
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
            go.Scattergl(x=labels, y=line_breakdown, fill="tozeroy", hoverinfo="x+y"),
            1,
            1,
        )
        # Bar charts showing the number of lines from each source
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[0.0]
                + [len(source) for source in [artifacts, splat, public, private]],
                hoverinfo="x+y",
                width=0.5,
                marker={"color": colors},
            ),
            1,
            1,
        )
        # Right column plot of the intensity contributions
        fig.add_trace(
            go.Scattergl(
                x=labels, y=intensity_breakdown, fill="tozeroy", hoverinfo="x+y"
            ),
            1,
            2,
        )
        # Bar chart showing the intensity contribution from each source
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[0.0]
                + [
                    np.sum(source["intensity"])
                    for source in [artifacts, splat, public, private]
                ],
                hoverinfo="x+y",
                width=0.5,
                marker={"color": colors},
            ),
            1,
            2,
        )
        fig["layout"]["xaxis1"].update(title="Source", showgrid=True)
        fig["layout"]["yaxis1"].update(
            title="Cumulative number of assignments", range=[0.0, total_lines * 1.05]
        )
        fig["layout"]["xaxis2"].update(title="Source", showgrid=True)
        fig["layout"]["yaxis2"].update(
            title="Cumulative intensity", range=[0.0, total_intensity * 1.05]
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
        fig.layout["title"] = f"Experiment {self.session.experiment}"
        fig.layout["xaxis"]["title"] = "Frequency (MHz)"
        fig.layout["xaxis"]["tickformat"] = ",.2f"

        # Update the peaks table
        self.peaks = pd.DataFrame(
            data=[
                [uline.frequency, uline.intensity]
                for uline in self.line_lists["Peaks"].get_ulines()
            ],
            columns=["Frequency", "Intensity"],
        )

        fig.add_scattergl(
            x=self.data["Frequency"],
            y=self.data[self.int_col],
            name="Experiment",
            opacity=0.6,
        )

        fig.add_bar(
            x=self.table["catalog_frequency"],
            y=self.table["intensity"],
            width=1.0,
            hoverinfo="text",
            text=self.table["name"].astype(str)
            + "-"
            + self.table["r_qnos"].astype(str),
            name="Assignments",
        )
        ulines = np.array(
            [
                [uline.intensity, uline.frequency]
                for uline in self.line_lists["Peaks"].get_ulines()
            ]
        )

        fig.add_bar(x=ulines[:, 1], y=ulines[:, 0], width=1.0, name="U-lines")
        return fig

    def stacked_plot(self, frequencies: List[float], int_col=None, freq_range=0.05):
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
        if int_col is None:
            int_col = self.int_col
        # Update the peaks table
        ulines = self.line_lists["Peaks"].get_ulines()
        self.peaks = pd.DataFrame(
            data=[[uline.frequency, uline.intensity] for uline in ulines],
            columns=["Frequency", "Intensity"],
        )
        dataframe = self.data.copy()
        # Want the frequencies in ascending order, going upwards in the plot
        indices = np.where(
            np.logical_and(
                dataframe[self.freq_col].min() <= frequencies,
                frequencies <= dataframe[self.freq_col].max(),
            )
        )
        # Plot only frequencies within band
        frequencies = np.asarray(frequencies)[indices]
        # Sort frequencies such that plots are descending in frequency
        frequencies = np.sort(frequencies)[::-1]
        delta = min(frequencies) * freq_range
        nplots = len(frequencies)
        if nplots > 5:
            raise Exception("Too many requested frequencies; I can't stack them all!")
        titles = tuple(f"{frequency:,.0f} MHz" for frequency in frequencies)
        fig = figurefactory.init_plotly_subplot(
            nrows=nplots,
            ncols=1,
            **{
                "subplot_titles": titles,
                "vertical_spacing": 0.15,
                "shared_xaxes": True,
            },
        )
        for index, frequency in enumerate(frequencies):
            # Calculate the offset frequency
            dataframe["Offset " + str(index)] = dataframe[self.freq_col] - frequency
            # Range as a fraction of the center frequency
            freq_cutoff = freq_range * frequency
            max_freq = frequency + freq_cutoff
            min_freq = frequency - freq_cutoff
            sliced_df = dataframe.loc[
                dataframe[f"Offset {index}"].between(-delta, delta)
            ]
            sliced_peaks = self.peaks.loc[
                self.peaks["Frequency"].between(min_freq, max_freq)
            ]
            sliced_peaks["Offset Frequency"] = sliced_peaks["Frequency"] - frequency
            sliced_assignments = self.table.loc[
                self.table["frequency"].between(min_freq, max_freq)
            ]
            sliced_assignments["offset_frequency"] = (
                sliced_assignments["frequency"] - frequency
            )
            # Plot the data
            traces = list()
            # Spectrum plot
            traces.append(
                go.Scattergl(
                    x=sliced_df["Offset " + str(index)],
                    y=sliced_df[self.int_col],
                    mode="lines",
                    opacity=0.6,
                    marker={"color": "rgb(5,113,176)"},
                )
            )
            traces.append(
                go.Bar(
                    x=sliced_assignments["offset_frequency"],
                    y=sliced_assignments["intensity"],
                    width=1.0,
                    hoverinfo="text",
                    text=sliced_assignments["name"]
                    + "-"
                    + sliced_assignments["r_qnos"].apply(str),
                    name="Assignments",
                    marker={"color": "rgb(253,174,97)"},
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
                    text=sliced_peaks["Frequency"],
                )
            )
            # Plotly indexes from one because they're stupid
            fig.add_traces(traces, [index + 1] * 3, [1] * 3)
            fig["layout"]["xaxis1"].update(
                range=[-freq_cutoff, freq_cutoff],
                title="Offset frequency (MHz)",
                showgrid=True,
            )
            fig["layout"]["yaxis" + str(index + 1)].update(showgrid=False)
        fig["layout"].update(autosize=True, height=1000, width=900, showlegend=False)
        return fig

    def match_artifacts(self, artifact_exp: "AssignmentSession", threshold=0.05):
        """
        TODO: Need to update this method; `process_artifacts` is no longer a method.
        
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
            self.logger.info(f"Removed {freq:,.4f} peak as artifact.")

    def find_progressions(
        self,
        search=0.001,
        low_B=400.0,
        high_B=9000.0,
        refit=False,
        plot=True,
        preferences=None,
        **kwargs,
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
            uline_frequencies, search=search, low_B=low_B, high_B=high_B
        )
        self.logger.info("Fitting progressions.")
        fit_df, _ = fitting.harmonic_fitter(progressions, J_thres=search)
        self.logger.info(f"Found {len(fit_df)} possible progressions.")
        pref_test_data = dict()
        # Run cluster analysis on the results
        if preferences is None:
            preferences = np.array(
                [-8e4, -4e4, -2e4, -1e4, -5e3, -3e3, -1.5e3, -500.0, -100.0]
            )
        # Perform the AP cluster modelling
        if type(preferences) == list or type(preferences) == np.ndarray:
            self.logger.info(f"Evaluating AP over grid values {preferences}.")
            for preference in preferences:
                try:
                    ap_settings = {"preference": preference}
                    ap_settings.update(**kwargs)
                    cluster_dict, progressions, _ = analysis.cluster_AP_analysis(
                        fit_df, sil_calc=True, refit=refit, **ap_settings
                    )
                    npoor = (progressions.loc[progressions["Silhouette"] < 0.0].size,)
                    if type(npoor) == tuple:
                        npoor = npoor[0]
                    nclusters = len(cluster_dict)
                    pref_test_data[preference] = {
                        "cluster_data": cluster_dict,
                        "nclusters": nclusters,
                        "npoor": npoor,
                        "avg_silh": np.average(progressions["Silhouette"]),
                        "progression_df": progressions,
                    }
                    self.logger.info(
                        f"{preference} has {npoor} poor fits "
                        f"out of {nclusters} clusters ({npoor / nclusters})."
                    )
                except ValueError:
                    pass
        else:
            # Perform the AP clustering with a single preference value
            cluster_dict, progressions, _ = analysis.cluster_AP_analysis(
                fit_df, sil_calc=True, refit=refit, **{"preference": preferences}
            )
            npoor = (progressions.loc[progressions["Silhouette"] < 0.0].size,)
            if type(npoor) == tuple:
                npoor = npoor[0]
            nclusters = len(cluster_dict)
            pref_test_data[preferences] = {
                "cluster_data": cluster_dict,
                "nclusters": nclusters,
                "npoor": npoor,
                "avg_silh": np.average(progressions["Silhouette"]),
                "progression_df": progressions,
            }
            self.logger.info(
                f"{preferences} has {npoor} poor fits "
                f"out of {nclusters} clusters ({npoor / nclusters})."
            )
        # Make a plotly figure of how the clustering goes (with frequency)
        # as a function of preference
        if plot is True:
            fig = go.FigureWidget()
            fig.layout["xaxis"]["title"] = "Frequency (MHz)"
            fig.layout["yaxis"]["title"] = "Preference"
            fig.layout["xaxis"]["tickformat"] = ".,"
            # Loop over the preference values
            for preference, contents in pref_test_data.items():
                # Create the colors for the unique clusters
                colors = figurefactory.generate_colors(
                    len(contents["cluster_data"]), hex=True, cmap=plt.cm.Spectral
                )
                # Assign a color to each cluster
                cmap = {
                    index: color
                    for index, color in zip(
                        np.arange(len(contents["cluster_data"])), colors
                    )
                }
                for index, data in contents["cluster_data"].items():
                    frequencies = data["Frequencies"]
                    fig.add_scattergl(
                        x=frequencies,
                        y=[preference + index] * len(frequencies),
                        mode="markers",
                        marker={"color": cmap.get(index, "#ffffff")},
                        opacity=0.7,
                        name=f"{preference}-{index}",
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
            filepath = f"./sessions/{self.session.experiment}.pkl"
        self.logger.info(f"Saving session to {filepath}")
        if hasattr(self, "log_handlers"):
            del self.log_handlers
        # Save to disk
        routines.save_obj(self, filepath)


@dataclass
class Molecule(LineList):
    """
    Special instance of the LineList class. The idea is to eventually
    use the high speed fitting/cataloguing routines by Brandon to provide
    quick simulations overlaid on chirp spectra.
    
    Attributes
    """

    A: float = 20000.0
    B: float = 6000.0
    C: float = 3500.0
    var_file: str = ""

    def __post_init__(self, **kwargs):
        super().__init__(**kwargs)

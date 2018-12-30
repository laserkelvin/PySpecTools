"""
    assignment.py

    Contains dataclass routines for tracking assignments
    in broadband spectra.
"""

import os
from shutil import rmtree
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd
from lmfit.models import GaussianModel
from IPython.display import display, HTML
from periodictable import formula
from plotly import graph_objs as go

from pyspectools import routines, parsers
from pyspectools import fitting
from pyspectools.spectra import analysis


@dataclass
class Assignment:
    """
        DataClass for handling assignments.
        There should be sufficient information to store a
        line assignment and reproduce it later in a form
        that is both machine and human readable.

        parameters:
        ------------------
        name - str representing IUPAC/common name
        formula - str representing chemical formula
        smiles - str representing SMILES code (specific!)
        frequency - float for observed frequency
        intensity - float for observed intensity
        peak_id - int for peak id from experiment
        uline - bool flag to signal whether line is identified or not
        composition - list-like with string corresponding to experimental
                      chemical composition. SMILES syntax.
        v_qnos - list with quantum numbers for vibrational modes. Index
                 corresponds to mode, and int value to number of quanta.
                 Length should be equal to 3N-6.
        experiment - int for experiment ID
        fit - dict-like containing the fitted parameters and model
    """
    name: str = ""
    smiles: str = ""
    formula: str = ""
    frequency: float = 0.0
    catalog_frequency: float = 0.0
    intensity: float = 0.0
    peak_id: int = 0
    experiment: int = 0
    uline: bool = True
    composition: List[str] = field(default_factory = list)
    v_qnos: List[int] = field(default_factory = list)
    r_qnos: str = ""
    fit: Dict = field(default_factory = dict)
    ustate_energy: float = 0.0
    interference: bool = False
    weighting: float = 0.0
    source: str = "Catalog"

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
        return f"{self.name}, {self.frequency}"

    def to_file(self, filepath, format="yaml"):
        """ Method to dump data to YAML format.
            Extensions are automatically decided, but
            can also be supplied.

            parameters:
            --------------------
            filepath - str path to yaml file
            format - str denoting the syntax used for dumping.
                     Defaults to YAML.
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

    def get_spectrum(self, x):
        """ Generate a synthetic peak by supplying
            the x axis for a particular spectrum. This method
            assumes that some fit parameters have been determined
            previously.

            parameters:
            ----------------
            x - 1D array with frequencies of experiment

            returns:
            ----------------
            y - 1D array of synthetic Gaussian spectrum
        """
        model = GaussianModel()
        params = model.make_params()
        params.update(self.fit)
        y = model.eval(params, x=x)
        return y

    @classmethod
    def from_dict(obj, data_dict):
        """ Method for generating an Assignment object
            from a dictionary. All this method does is
            unpack a dictionary into the __init__ method.

            parameters:
            ----------------
            data_dict - dict with DataClass fields

            returns:
            ----------------
            Assignment object
        """
        assignment_obj = obj(**data_dict)
        return assignment_obj

    @classmethod
    def from_yml(obj, yaml_path):
        """ Method for initializing an Assignment object
            from a YAML file.

            parameters:
            ----------------
            yaml_path - str path to yaml file

            returns:
            ----------------
            Assignment object
        """
        yaml_dict = routines.read_yaml(yaml_path)
        assignment_obj = obj(**yaml_dict)
        return assignment_obj

    @classmethod
    def from_json(obj, json_path):
        """ Method for initializing an Assignment object
            from a JSON file.

            parameters:
            ----------------
            json_path - str path to json file

            returns:
            ----------------
            Assignment object
        """
        json_dict = routines.read_json(json_path)
        assignment_obj = obj(**json_dict)
        return assignment_obj


@dataclass
class Session:
    """ DataClass for a Session, which simply holds the
        experiment ID, composition, and guess temperature
    """
    experiment: int
    composition: List[str] = field(default_factory = list)
    temperature: float = 4.0


class AssignmentSession:
    """ Class for managing a session of assigning molecules
        to a broadband spectrum.

        Wraps some high level functionality from the analysis
        module so that this can be run reproducibly in a jupyter
        notebook.
    """

    @classmethod
    def load_session(cls, filepath):
        """
            Load an AssignmentSession from disk, once it has
            been saved with the save_session method.

            parameters:
            --------------
            filepath - path to the AssignmentSession file; typically
                       in the sessions/{experiment_id}.dat
        """
        session = routines.read_obj(filepath)
        obj = cls(
            session["data"],
            **session["session"]
            )
        # Reinitialize classes
        obj.ulines = [
            Assignment.from_dict(uline) for uline in session["ulines"]
            ]
        obj.assignments = [
            Assignment.from_dict(ass_obj) for ass_obj in session["assignments"]
            ]
        return obj 

    def __init__(
            self, exp_dataframe, experiment, composition, temperature=4.0,
            freq_col="Frequency", int_col="Intensity"):
        """ Initialize a AssignmentSession with a pandas dataframe
            corresponding to a spectrum.

            description of data:
            -------------------------
            exp_dataframe - pandas dataframe with observational data in frequency/intensity
            experiment - int ID for the experiment
            composition - list of str corresponding to experiment composition
            freq_col - optional arg specifying the name for the frequency column
            int_col - optional arg specifying the name of the intensity column
        """
        # Make folders for organizing output
        folders = ["assignment_objs", "queries", "sessions", "clean", "reports"]
        for folder in folders:
            if os.path.isdir(folder) is False:
                os.mkdir(folder)
        # Initialize a Session dataclass
        self.session = Session(experiment, composition, temperature)
        self.data = exp_dataframe
        self.t_threshold = self.session.temperature * 3.
        self.assignments = list()
        self.ulines = list()
        # Default settings for columns
        if freq_col not in self.data.columns:
            self.freq_col = self.data.columns[0]
        else:
            self.freq_col = freq_col
        if int_col not in self.data.columns:
            self.int_col = self.data.columns[1]
        else:
            self.int_col = int_col

    def find_peaks(self, threshold=0.015):
        """ Wrap peakutils method for detecting peaks.

            parameters:
            ---------------
            threshold - peak detection threshold

            returns:
            ---------------
            peaks_df - dataframe containing peaks
        """
        peaks_df = analysis.peak_find(
            self.data,
            freq_col=self.freq_col,
            int_col=self.int_col,
            thres=threshold
           )
        # Reindex the peaks
        peaks_df.index = np.arange(len(peaks_df))
        if hasattr(self, "peaks") is True:
            # If we've looked for peaks previously
            # we don't have to re-add the U-line to
            # the list
            peaks_df = pd.concat([peaks_df, self.peaks])
            # drop repeated frequencies
            peaks_df.drop_duplicates(["Frequency"],inplace=True)
        # Generate U-lines
        selected_session = {
            key: self.session.__dict__[key] for key in self.session.__dict__ if key != "temperature"
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
            if ass_obj not in self.ulines and ass_obj not in self.assignments:
                self.ulines.append(ass_obj)
        # Assign attribute
        self.peaks = peaks_df
        return peaks_df

    def search_frequency(self, frequency):
        """ Method for searching a frequency in the spectrum.
            Gives information relevant to whether it's a U-line
            or assigned.

            Raises exception error if nothing is found.

            parameters:
            ----------------
            frequency - float corresponding to the frequency in MHz

            returns:
            ----------------
            slice_df - pandas dataframe containing matches
        """
        slice_df = []
        lower_freq = frequency * 0.999
        upper_freq = frequency * 1.001
        if hasattr(self, "table"):
            slice_df = self.table.loc[
                (self.table["Frequency"] >= lower_freq) & 
                (self.table["Frequency"] <= upper_freq)
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
        """ Method to ask a simple yes/no if the frequency
            exists in either U-lines or assignments.

            parameters:
            ---------------
            frequency - float corresponding to frequency in MHz

            returns:
            --------------
            bool - True if it's in ulines/assignments, False otherwise
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
        """ Function that will provide an "interface" for interactive
            line assignment in a notebook environment.

            Basic functionality is looping over a series of peaks,
            which will query splatalogue for known transitions in the
            vicinity. If the line is known in Splatalogue, it will
            throw it into an Assignment object and flag it as known.
            Conversely, if it's not known in Splatalogue it will defer
            assignment, flagging it as unassigned and dumping it into
            the `uline` attribute.
        """
        if hasattr(self, "peaks") is False:
            print("Peak detection not run; running with default settings.")
            self.find_peaks()

        for uindex, uline in enumerate(self.ulines):
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

            splat_df = self.calc_line_weighting(frequency, splat_df, prox=0.01)
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
                       "formula": ass_df["Species"][0],
                       "r_qnos": ass_df["Resolved QNs"][0],
                       "ustate_energy": ass_df["E_U (K)"][0],
                       "weighting": ass_df["Weighting"][0],
                       "source": "CDMS/JPL"
                       }
                    # Perform a Voigt profile fit
                    print("Attempting to fit line profile...")
                    fit_results = analysis.fit_line_profile(
                        self.data,
                        frequency
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

            Kwargs are passed into the assignment dictionary.

            parameters:
            -----------------
            name - str common name of the molecule
            formula - str chemical formula of the molecule
            linpath - path to line file to be parsed
            auto - optional bool specifying which mode to run in
        """
        old_nulines = len(self.ulines)
        lin_df = parsers.parse_lin(linpath)

        for uindex, uline in enumerate(self.ulines):
            sliced_catalog = self.calc_line_weighting(
                uline.frequency,
                lin_df
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
                        "index": uindex,
                        "frequency": uline.frequency,
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "source": "Line file"
                        }
                    assign_dict.update(**kwargs)
                    self.assign_line(**assign_dict)
        ass_df = pd.DataFrame(
            data=[ass_obj.__dict__ for ass_obj in self.assignments]
            )
        self.table = ass_df
        print("Prior number of ulines: {}".format(old_nulines))
        print("Current number of ulines: {}".format(len(self.ulines)))


    def calc_line_weighting(self, frequency, catalog_df, prox=0.0001):
        """
            Function for calculating the weighting factor for determining
            the likely hood of an assignment. The weighting factor is
            determined by the proximity of the catalog frequency to the
            observed frequency, as well as the theoretical intensity if it
            is available.

            parameters:
            ----------------
            frequency - float observed frequency in MHz
            catalog_df - dataframe corresponding to catalog entries
            prox - optional float for frequency proximity threshold

            returns:
            ---------------
            If nothing matches the frequency, returns None.
            If matches are found, calculate the weights and return the candidates
            in a dataframe.
        """
        lower_freq = frequency * (1 - prox)
        upper_freq = frequency * (1 + prox)
        sliced_catalog = catalog_df.loc[
            (catalog_df["Frequency"] >= lower_freq) & (catalog_df["Frequency"] <= upper_freq)
            ]
        nentries = len(sliced_catalog)
        if nentries > 0:
            # Calculate probability weighting. Base is the inverse of distance
            sliced_catalog["Deviation"] = np.abs(sliced_catalog["Frequency"] - frequency)
            sliced_catalog["Weighting"] = (1./sliced_catalog["Deviation"])
            # If intensity is included in the catalog incorporate it in
            # the weight calculation
            if "Intensity" in sliced_catalog:
                sliced_catalog["Weighting"]*=(10**sliced_catalog["Intensity"])
            elif "CDMS/JPL Intensity" in sliced_catalog:
                sliced_catalog["Weighting"]*=(10**sliced_catalog["CDMS/JPL Intensity"])
            else:
                # If there are no recognized intensity columns, pass
                pass
            # Normalize the weights
            sliced_catalog["Weighting"]/=sliced_catalog["Weighting"].max()
            # Sort by obs-calc
            sliced_catalog.sort_values(["Weighting"], ascending=True, inplace=True)
            sliced_catalog.index = np.arange(nentries)
            return sliced_catalog
        else:
            return None

    def process_catalog(self, name, formula, catalogpath, auto=True, **kwargs):
        """
            Reads in a catalog (SPCAT) file and sweeps through the
            U-line list for this experiment finding coincidences.

            Similar to the splatalogue interface, lines are rejected
            on state energy from the t_threshold attribute.

            Each catalog entry will be weighted according to their
            theoretical intensity and deviation from observation. In
            automatic mode, the highest weighted transition is assigned.

            parameters:
            ----------------
            name - str corresponding to common name of molecule
            formula - str corresponding to chemical formula
        """
        old_nulines = len(self.ulines)
        catalog_df = parsers.parse_cat(
                catalogpath,
                self.data[self.freq_col].min(),
                self.data[self.freq_col].max()
            )
        # Filter out the states with high energy
        catalog_df = catalog_df.loc[
            catalog_df["Lower state energy"] <= self.t_threshold
            ]
        # Loop over the uline list
        for uindex, uline in enumerate(self.ulines):
            # 0.1% of frequency
            sliced_catalog = self.calc_line_weighting(uline.frequency, catalog_df)
            if sliced_catalog is not None:
                display(HTML(sliced_catalog.to_html()))
                if auto is False:
                    index = int(raw_input("Please choose a candidate by index."))
                elif auto is True:
                    index = 0
                if index in sliced_catalog.index:
                    select_df = sliced_catalog.iloc[index]
                    # Create an approximate quantum number string
                    qnos = "N'={}, J'={}".format(*sliced_catalog[["N'", "J'"]].values[0])
                    qnos+="N''={}, J''={}".format(*sliced_catalog[["N''", "J''"]].values[0])
                    assign_dict = {
                        "name": name,
                        "formula": formula,
                        "index": uindex,
                        "frequency": uline.frequency,
                        "r_qnos": qnos,
                        "catalog_frequency": select_df["Frequency"],
                        "source": "Catalog"
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

            parameters:
            -----------------
            name - str denoting the name of the molecule
            index - optional arg specifying U-line index
            frequency - optional float specifying frequency to assign
            **kwargs - passed to update Assignment object
        """
        if index == frequency:
            raise Exception("Index/Frequency not specified!")
        ass_obj = None
        if index:
            # If an index is supplied, pull up from uline list
            ass_obj = self.ulines[index]
        elif frequency:
            for index, obj in enumerate(self.ulines):
                deviation = np.abs(frequency - obj.frequency)
                # Check that the deviation is sufficiently small
                if deviation <= (frequency * 1e-5):
                    # Remove from uline list
                    ass_obj = obj
                    frequency = ass_obj.frequency
        if ass_obj:
            ass_obj.name = name
            ass_obj.uline = False
            # Unpack anything else
            ass_obj.__dict__.update(**kwargs)
            if frequency is None:
                frequency = ass_obj.frequency
            ass_obj.frequency = frequency
            print("{:,.4f} assigned to {}".format(frequency, name))
            self.ulines.pop(index)
            self.assignments.append(ass_obj)
        else:
            raise Exception("Peak not found! Try providing an index.")

    def get_assigned_names(self):
        """ Method for getting all the unique molecules out
            of the assignments, and tally up the counts.
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

    def finalize_assignments(self):
        """
            Function that will complete the assignment process by
            serializing DataClass objects and formatting a report.
        """
        for ass_obj in self.assignments:
            # Dump all the assignments into YAML format
            ass_obj.to_file(
                "assignment_objs/{}-{}".format(ass_obj.experiment,ass_obj.peak_id),
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
        # Dump data to notebook output
        for key, value in combined_dict.items():
            print(key + ":   " + str(value))

    def clean_folder(self, action=False):
        """ Method for cleaning up all of the directories used by this routine.
            Use with caution!!!

            Requires passing a True statement to actually clean up.
        """
        folders = ["assignment_objs", "queries", "sessions", "clean", "reports"]
        if action is True:
            for folder in folders:
                rmtree(folder)

    def plot_assigned(self):
        """
            Generates a Plotly figure with the assignments overlaid
            on the experimental spectrum.
        """
        fig = go.FigureWidget()

        fig.add_scatter(
            x=self.data["Frequency"],
            y=self.data["Intensity"],
            name="Experiment",
            opacity=0.4
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
        # Store as attribute
        self.plot = fig

        return fig

    def find_progressions(self, search=0.001, low_B=400.,
            high_B=9000., sil_calc=True, refit=False, plot=True, **kwargs):
        """
            High level function for searching U-line database for
            possible harmonic progressions. Wraps the lower level function
            harmonic_search, which will generate 3-4 frequency combinations
            to search.

            parameters:
            ---------------
            search - threshold for determining if quantum number J is close
                     enough to an integer/half-integer
            low_B - minimum value for B
            high_B - maximum value for B
            plot - whether or not to produce a plot of the progressions

            returns:
            ---------------
            harmonic_df - dataframe containing the viable transitions
        """
        uline_frequencies = [uline.frequency for uline in self.ulines]

        progressions = analysis.harmonic_finder(
            uline_frequencies,
            search=search,
            low_B=low_B,
            high_B=high_B
            )

        fit_df = fitting.harmonic_fitter(progressions)

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

            parameters:
            --------------
            formula - str for chemical formula lookup
            name - str for common name
            smiles - str for unique SMILES string

            returns:
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
            and so there are can cross-compatibility issues particularly
            when loading from different versions of Python.

            parameters:
            ---------------
            filepath - path to save the file to. By default it will go into
                       the sessions folder.
        """
        if filepath is None:
            filepath = "./sessions/{}.dat".format(self.session.experiment)

        # Unpack everything so that object serialization works properly...
        session_dict = self.__dict__
        session_dict["session"] = self.session.__dict__
        session_dict["ulines"] = [uline.__dict__ for uline in self.ulines]
        session_dict["assignments"] = [ass_obj.__dict__ for ass_obj in self.assignments]
        # Save to disk
        routines.save_obj(
            self.__dict__,
            filepath
            )
        print("Saved session to {}".format(filepath))


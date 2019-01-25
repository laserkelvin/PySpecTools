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
from plotly.offline import plot
from plotly import graph_objs as go

from pyspectools import routines, parsers, figurefactory
from pyspectools import fitting
from pyspectools import units
from pyspectools.astro import analysis as aa
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
        I - float for theoretical intensity
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
        return f"{self.name}, {self.frequency}"

    def to_file(self, filepath, format="yaml"):
        """ Method to dump data to YAML format.
            Extensions are automatically decided, but
            can also be supplied.

            parameters:
            --------------------
            :param filepath: str path to yaml file
            :param format: str denoting the syntax used for dumping.
                     Defaults to YAML.
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
        experiment ID, composition, and guess_temperature.
        Doppler broadening can also be incorporated. 

        parameters:
        --------------
        experiment: integer ID for experiment
        composition: list of strings corresponding to atomic
                     symbols
        temperature: float temperature
        doppler: float doppler in km/s; default value is about
                 5 kHz at 15 GHz.
    """
    experiment: int
    composition: List[str] = field(default_factory=list)
    temperature: float = 4.0
    doppler: float = 0.01

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
        return session

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

    def umol_gen(self):
        """
        Method for keeping track of what unidentified molecule
        we're up to
        :return: generator for a formatted string to name unidentified molecules
        """
        counter = 1
        while counter <= 200:
            yield "UMol_{:03.d}".format(counter)
            counter+=1

    def find_peaks(self, threshold=None):
        """ Wrap peakutils method for detecting peaks.

            parameters:
            ---------------
            :param threshold: peak detection threshold

            returns:
            ---------------
            :return peaks_df: dataframe containing peaks
        """
        if threshold is None:
            # Set the threshold as 20% of the baseline + 1sigma. The peak_find function will
            # automatically weed out the rest.
            threshold = (self.data[self.int_col].mean() + self.data[self.int_col].std()) * 0.2
            print("Peak detection threshold is: {}".format(threshold))
        peaks_df = analysis.peak_find(
            self.data,
            freq_col=self.freq_col,
            int_col=self.int_col,
            thres=threshold
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
        skip = ["temperature", "doppler"]
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
            :param frequency: float corresponding to the frequency in MHz

            returns:
            ----------------
            :return slice_df: pandas dataframe containing matches
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
            :param frequency: float corresponding to frequency in MHz

            returns:
            --------------
            :return bool: True if it's in ulines/assignments, False otherwise
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

            Kwargs are passed into the assignment dictionary.

            parameters:
            -----------------
            :param name: str common name of the molecule
            :param formula: str chemical formula of the molecule
            :param linpath: path to line file to be parsed
            :param auto: optional bool specifying which mode to run in
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

    def calc_line_weighting(self, frequency, catalog_df, prox=0.0001):
        """
            Function for calculating the weighting factor for determining
            the likely hood of an assignment. The weighting factor is
            determined by the proximity of the catalog frequency to the
            observed frequency, as well as the theoretical intensity if it
            is available.

            parameters:
            ----------------
            :param frequency: float observed frequency in MHz
            :param catalog_df: dataframe corresponding to catalog entries
            :param prox: optional float for frequency proximity threshold

            returns:
            ---------------
            :returns If nothing matches the frequency, returns None.
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
            sliced_catalog["Weighting"] = (1. / sliced_catalog["Deviation"])
            # If intensity is included in the catalog incorporate it in
            # the weight calculation
            if "Intensity" in sliced_catalog:
                sliced_catalog["Weighting"] *= (10 ** sliced_catalog["Intensity"])
            elif "CDMS/JPL Intensity" in sliced_catalog:
                sliced_catalog["Weighting"] *= (10 ** sliced_catalog["CDMS/JPL Intensity"])
            else:
                # If there are no recognized intensity columns, pass
                pass
            # Normalize the weights
            sliced_catalog["Weighting"] /= sliced_catalog["Weighting"].max()
            # Sort by obs-calc
            sliced_catalog.sort_values(["Weighting"], ascending=False, inplace=True)
            sliced_catalog.reset_index(drop=True, inplace=True)
            return sliced_catalog
        else:
            return None

    def process_catalog(self, name, formula, catalogpath, 
            auto=True, thres=-10., **kwargs):
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
            :param name: str corresponding to common name of molecule
            :param formula: str corresponding to chemical formula
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
        for uindex, uline in enumerate(self.ulines):
            # 0.1% of frequency
            sliced_catalog = self.calc_line_weighting(uline.frequency, catalog_df)
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
                        "index": uindex,
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

    def process_frequencies(self, frequencies, ids, molecule=None):
        """
        Function to mark frequencies to belong to a single molecule, and for book-keeping's
        sake a list of ids are also required to indicate the original scan as the source
        of the information.

        :param frequencies: list of frequencies associated with a molecule
        :param ids: list of scan IDs
        :param molecule: optional str specifying the name of the molecule
        """
        counter = 0
        for freq, scan_id in zip(frequencies, ids):
            uline_freqs = np.array([uline.frequency for uline in self.ulines])
            nearest, index = routines.find_nearest(uline_freqs, freq)
            # Find the nearest U-line, and if it's sufficiently close then
            # we assign it to this molecule. This is just to make sure that
            # if we sneak in some random out-of-band frequency this we won't
            # just assign it
            if np.abs(nearest - freq) <= 0.2:
                assign_dict = {
                    "name": molecule,
                    "source": "Scan-{}".format(scan_id),
                    "catalog_frequency": freq,
                    "index": index,
                    "frequency": nearest
                }
                self.assign_line(**assign_dict)
                counter+=1
            else:
                print("No U-line was sufficiently close.")
                print("Expected: {}, Nearest: {}".format(freq, nearest))
        print("Tentatively assigned {} lines to {}.".format(counter, molecule))

    def verify_molecules(self):
        """
        Function that is run following the splatalogue assignment routine. This will
        go through all of the assigned species, and using the weakest line as basis,
        search for other possible transitions that have not already been assigned.

        In this mode, the search criterion for transitions are loosened w.r.t.
        upper state temperature and line intensity, as well as the peak finding
        algorithm.
        :return possible: dict containing dataframes for each molecule and transition
        """
        possible = dict()
        counter = 0
        ass_df = pd.DataFrame(
            [ass_obj.__dict__ for ass_obj in self.assignments]
        )
        # this gets a DataFrame with the highest upper state energies for each unique molecule
        high_df = ass_df.sort_values(["ustate_energy"], ascending=False).drop_duplicates(["name"])
        for mol_index, mol_row in high_df.iterrows():
            possible[mol_row["name"]] = dict()
            splat_df = analysis.search_molecule(
                mol_row["name"],
                [
                    self.data[self.freq_col].min(),
                    self.data[self.freq_col].max()
                ]
            )
            assigned_df = ass_df.loc[ass_df["name"] == mol_row["name"]]
            # Loosen the temperature and intensity criterion criterion
            filtered_df = splat_df.loc[
                (splat_df["E_U (K)"] <= mol_row["ustate_energy"] * 2.)
            ]
            if mol_row["catalog_intensity"] != 0.:
                # If catalog intensity was known previously, we'll also filter out the
                # intensities too
                filtered_df = filtered_df.loc[
                    filtered_df["CDMS/JPL Intensity"] >= mol_row["catalog_intensity"] * 1.5
                ]
            # Get only the transitions that haven't been assigned already
            mask = [freq not in assigned_df["catalog_frequency"].values for freq in filtered_df["Frequency"].values]
            filtered_df = filtered_df.loc[mask]
            """
                Loop over each molecular frequency, and slice up a bit of the observed spectrum to look for
                peaks again.
            """
            for index, row in filtered_df.iterrows():
                frequency = row["Frequency"]
                min_freq = frequency - 4.
                max_freq = frequency + 4.
                spectrum_slice = self.data.loc[
                    (self.data[self.freq_col] >= min_freq) &
                    (self.data[self.freq_col] <= max_freq)
                ]
                spectrum_slice.reset_index(inplace=True, drop=True)
                # Find peaks in the region of interest with a dynamic threshold for detection
                peaks = analysis.peak_find(
                    spectrum_slice, self.freq_col, self.int_col, thres=spectrum_slice[self.int_col].max() * 0.2
                )
                if len(peaks) > 0:
                    print("Found {} peaks between {} and {} MHz.".format(len(peaks), min_freq, max_freq))
                    # Remove peaks that have already been picked up in ulines/assigned
                    assigned_check = [peak_freq not in ass_df["frequency"] for peak_freq in peaks["Frequency"].values]
                    # If the lines were in any of the lists, they would return True. We want lines that
                    # aren't in the lists
                    peaks = peaks.loc[assigned_check]
                    peaks["Distance"] = np.abs(peaks["Frequency"] - row["Frequency"])
                    viable_freq = peaks["Frequency"].ix[peaks["Distance"].idxmin()]
                    viable_int = peaks["Intensity"].ix[peaks["Distance"].idxmin()]
                    uline_check = [uline_obj for  uline_obj in self.ulines if uline_obj.frequency == frequency]
                    # If we already had a U-line of this frequency we'll mark it as assigned
                    # Set up the assignment fields
                    ass_dict = {
                        "uline": False,
                        "frequency": frequency,
                        "intensity": viable_int,
                        "name": mol_row["name"],
                        "catalog_frequency": viable_freq,
                        "catalog_intensity": row["CDMS/JPL Intensity"],
                        "formula": row["Species"],
                        "r_qnos": row["Resolved QNs"],
                        "ustate_energy": row["E_U (K)"],
                        "source": "Post-CDMS/JPL",
                        "deviation": frequency - viable_freq
                    }
                    if len(uline_check) > 0:
                        ass_dict["index"] = uline_check[0].peak_id
                        self.assign_line(**ass_dict)
                    else:
                        # We create the assignment object directly because the U-line may not have
                        # already been there
                        self.assignments.append(
                            Assignment(**ass_dict)
                        )
                    print("Assigned ")
                    counter+=1
                possible[mol_row["name"]][frequency] = peaks
        print("Assigned a total of {} peaks".format(counter))
        return possible

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
            :param name: str denoting the name of the molecule
            :param index: optional arg specifying U-line index
            :param frequency: optional float specifying frequency to assign
            :param kwargs: passed to update Assignment object
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

            returns:
            ---------------
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

            parameters:
            --------------
            :param Q - rotational partition function at temperature
            :param T - temperature in K
            :param name - str name of molecule
            :param formula - chemical formula of molecule
            :param smiles - SMILES string for molecule

            returns:
            --------------
            :return profile_df - pandas dataframe containing all of the
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

    def clean_folder(self, action=False):
        """
            Method for cleaning up all of the directories used by this routine.
            Use with caution!!!

            Requires passing a True statement to actually clean up.

            :param action: bool; will only clean up when True is passed
        """
        folders = ["assignment_objs", "queries", "sessions", "clean", "reports"]
        if action is True:
            for folder in folders:
                rmtree(folder)

    def simulate_sticks(self, catalogpath, N, Q, T, doppler=None, gaussian=False):
        """
        Simulates a stick spectrum with intensities in flux units (Jy) for
        a given catalog file, the column density, and the rotational partition
        function at temperature T.
        :param catalogpath: path to SPCAT catalog file
        :param N: column density in cm^-2
        :param Q: partition function at temperature T
        :param T: temperature in Kelvin
        :param doppler: doppler width in km/s; defaults to session wide value
        :param gaussian: bool; if True, simulates Gaussian profiles instead of sticks
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

            parameters:
            ---------------
            x - array of x values to evaluate Gaussians on
            centers - array of Gaussian centers
            widths - array of Gaussian widths
            amplitudes - array of Gaussian amplitudes

            returns:
            ---------------
            y - array of y values
        """
        y = np.zeros(len(x))
        model = GaussianModel()
        for c, w, a in zip(centers, widths, amplitudes):
            if fake is True:
                scaling = a
            else:
                scaling = 1.
            y+=scaling * model.eval(
                x=x,
                center=c,
                sigma=w,
                amplitude=a
                )
        return y

    def plot_spectrum(self):
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
            centers = np.array([uline.frequency for uline in self.ulines])
            widths = units.dop2freq(
                self.session.doppler,
                centers
                )
            amplitudes = np.array([uline.intensity for uline in self.ulines])
            labels = [uline.peak_id for uline in self.ulines]

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

            fig.add_bar(
                x=centers,
                y=amplitudes,
                hoverinfo="text",
                text=labels,
                name="Peaks"
                )

        return fig

    def create_html_report(self, filepath=None):
        """
        Function for generating an HTML report for sharing.
        :param filepath: str path to save the report to. Defaults to reports/{id}-summary.html
        """
        from jinja2 import Template
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "report_template.html"
        )
        with open(template_path) as read_file:
            template = Template(read_file.read())
        html_dict = dict()
        # The assigned molecules table
        reduced_table = self.table[
            ["frequency", "intensity", "formula", "name", "catalog_frequency", "deviation", "ustate_energy", "source"]
        ]
        html_dict["assignments_table"] = reduced_table.to_html()
        # The unidentified features table
        uline_df = pd.DataFrame(
            [[uline.frequency, uline.intensity] for uline in self.ulines], columns=["Frequency", "Intensity"]
        )
        html_dict["uline_table"] = uline_df.to_html()
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
        Generate a pie chart to summarize the breakdown of spectral features.
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
        total_intensity = np.sum([uline.intensity for uline in self.ulines]) + np.sum(reduced_table["intensity"])
        line_breakdown = [total_lines]
        intensity_breakdown = [total_intensity]
        for source in [artifacts, splat, public, private]:
            line_breakdown.append(-len(source))
            intensity_breakdown.append(-np.sum(source["intensity"]))
        line_breakdown = np.cumsum(line_breakdown)
        intensity_breakdown = np.cumsum(intensity_breakdown)
        colors = ["#d7191c", "#fdae61", "#abdda4", "#2b83ba"]
        fig.add_trace(
            go.Scattergl(
                x=sources,
                y=line_breakdown,
                fill="tozeroy",
                hoverinfo="x+y"
            ),
            1,
            1
        )
        fig.add_trace(
            go.Bar(
                x=sources,
                y=[len(source) for source in [artifacts, splat, public, private]],
                hoverinfo="x+y",
                width=0.5,
                marker={"color": colors}
            ),
            1,
            1
        )
        fig.add_trace(
            go.Scattergl(
                x=sources,
                y=intensity_breakdown,
                fill="tozeroy",
                hoverinfo="x+y"
            ),
            1,
            2
        )
        fig.add_trace(
            go.Bar(
                x=sources,
                y=[np.sum(source["intensity"]) for source in [artifacts, splat, public, private]],
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
            filepath = "./sessions/{}.pkl".format(self.session.experiment)
        # Save to disk
        routines.save_obj(
            self,
            filepath
        )
        print("Saved session to {}".format(filepath))

"""
    database.py

    Routines for managing a spectral line database.

    TODO - set up routines for a persistent database
"""

import os
import warnings

try:
    import tables
    from tables import IsDescription, open_file
    from tables import StringCol, Int64Col, Float64Col
except ImportError:
    warnings.warn(f"PyTables is not installed correctly!")

import tinydb
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
import pandas as pd

from pyspectools import parsers
from pyspectools import spectra


class SpectralCatalog(tinydb.TinyDB):
    """
    Grand unified experimental catalog. Stores assignment and uline information
    across the board.
    """

    def __init__(self, dbpath=None):
        if dbpath is None:
            dbpath = os.path.expanduser("~/.pyspectools/pyspec_experiment.db")
        super().__init__(
            dbpath,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            storage=CachingMiddleware(JSONStorage),
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Dunder method that should be called when the object is destroyed. This will make sure
        the database is saved properly.
        """
        self.close()

    def add_entry(self, assignment, dup_check=True):
        """
        This function adds an Transition object to an existing database. The method will
        check for duplicates before adding.

        Parameters
        ----------
        assignment - Transition object
            Reference to an Transition object
        dup_check - bool, optional
            If True (default), will check to make sure the Transition object doesn't already exist in
            the database.
        """
        add = False
        if type(assignment) != dict:
            new_entry = assignment.__dict__
        else:
            new_entry = assignment
        if dup_check is True:
            if any([new_entry == entry for entry in self.all()]) is False:
                add = True
            else:
                warnings.warn("Entry already exists in database.")
        else:
            add = True
        if add is True:
            self.insert(new_entry)

    def add_catalog(self, catalog_path, name, formula, **kwargs):
        """
        Load a SPCAT catalog file into the database. Creates independent Transition objects
        from each line of the catalog file. Kwargs are passed into the Transition object,
        which will allow additional settings for the Transition object to be accessed.
        :param catalog_path:
        :param name:
        :param formula:
        :param kwargs:
        :return:
        """
        # check if the name and formula exists already
        exist_df = self.search_molecule(name)
        cat_df = parsers.parse_cat(catalog_path)
        if exist_df is not None:
            # drop all of the entries that are already in the catalog
            exist_freqs = exist_df["frequency"].values
            cat_df = cat_df.loc[~cat_df["Frequency"].isin(list(exist_freqs)),]
        assign_dict = {"name": name, "formula": formula}
        assign_dict.update(**kwargs)
        # slice out only the relevant information from the dataframe
        select_df = cat_df[["Frequency", "Intensity", "Lower state energy"]]
        select_df.columns = ["catalog_frequency", "catalog_intensity", "ustate_energy"]
        select_dict = select_df.to_dict(orient="records")
        # update each line with the common data entries
        assignments = [
            spectra.assignment.Transition(**line, **assign_dict).__dict__
            for line in select_dict
        ]
        # Insert all of the documents en masse
        self.insert_multiple(assignments)

    def search_frequency(self, frequency, freq_prox=0.1, freq_abs=True, dataframe=True):
        """\
        :param frequency: float, center frequency to search for in the database
        :param freq_prox: float, search range tolerance. If freq_abs is True, the absolute value is used (in MHz).
                          Otherwise, freq_prox is a decimal percentage of the frequency.
        :param freq_abs: bool, dictates whether the absolute value of freq_prox is used.
        :return:
        """
        frequency = float(frequency)
        if freq_abs is True:
            min_freq = frequency - freq_prox
            max_freq = frequency + freq_prox
        else:
            min_freq = frequency * (1 - freq_prox)
            max_freq = frequency * (1 + freq_prox)
        Entry = tinydb.Query()
        matches = self.search(
            (Entry["frequency"] <= max_freq) & (min_freq <= Entry["frequency"])
            | (Entry["catalog_frequency"] <= max_freq)
            & (min_freq <= Entry["catalog_frequency"])
        )
        if len(matches) != 0:
            if dataframe is True:
                return pd.DataFrame(matches)
            else:
                return matches
        else:
            return None

    def _search_field(self, field, value, dataframe=True):
        """
        Function for querying the database for a particular field and value.
        The option dataframe specifies whether the matches are returned as a
        pandas datafarame, or as a list of Transition objects.
        :param field: str field to query
        :param value: value to compare with
        :param dataframe: bool, if True will return the matches as a pandas dataframe.
        :return:
        """
        matches = self.search(tinydb.where(field) == value)
        if len(matches) != 0:
            if dataframe is True:
                df = pd.DataFrame(matches)
                return df
            else:
                objects = [spectra.transition.Transition(**data) for data in matches]
                return objects
        else:
            return None

    def search_molecule(self, name, dataframe=True):
        """
        Search for a molecule in the database based on its name (not formula!).
        Wraps the _search_field method, which will return None if nothing is found, or either a
        pandas dataframe or a list of Transition objects
        :param name: str, name (not formula) of the molecule to search for
        :param dataframe: bool, if True, returns a pandas dataframe
        :return: matches: a dataframe or list of Transition objects that match the search name
        """
        matches = self._search_field("name", name, dataframe)
        return matches

    def search_experiment(self, exp_id, dataframe=True):
        matches = self._search_field("experiment", exp_id, dataframe)
        return matches

    def search_formula(self, formula, dataframe=True):
        matches = self._search_field("formula", formula, dataframe)
        return matches

    def _remove_field(self, field, value):
        Entry = tinydb.Query()
        self.remove(Entry[field] == value)

    def remove_experiment(self, exp_id):
        """
        Remove all entries based on an experiment ID.
        :param exp_id: int, experiment ID
        """
        self._remove_field("exp_id", exp_id)

    def remove_molecule(self, name):
        self._remove_field("name", name)


class TheoryCatalog(tinydb.TinyDB):
    """
    Grand unified theory catalog.
    """

    def __init__(self, dbpath=None):
        if dbpath is None:
            dbpath = os.path.expanduser("~/.pyspectools/pyspec_theory.db")
        super().__init__(dbpath)

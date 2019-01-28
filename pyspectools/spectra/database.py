"""
    database.py

    Routines for managing a spectral line database.

    TODO - set up routines for a persistent database
"""

import os

import tinydb


class SpectralCatalog(tinydb.TinyDB):
    def __init__(self, dbpath=None):
        if dbpath is None:
            dbpath = os.path.expanduser("~/.pyspectools/pyspec.db")
        super().__init__(dbpath)

    def search_frequency(self, frequency):
        """
        TODO make a frequency look up function
        :param frequency:
        :return:
        """
        return None

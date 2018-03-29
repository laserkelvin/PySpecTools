
from lmfit import models
from astroquery.splatalogue import Splatalogue
from astropy import units as u


def search_center_frequency(frequency, width=0.5):
    """ Wrapper for the astroquery Splatalogue search
        This function will take a center frequency, and query splatalogue
        within the CDMS and JPL linelists for carriers of the line.

        Input arguments:
        frequency - float specifying the center frequency
    """
    table = Splatalogue.query_lines(
        frequency - width,
        frequency + width,
    )

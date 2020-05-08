.. PySpecTools documentation master file, created by
   sphinx-quickstart on Mon Nov 18 08:52:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reproducible, automated, and interactive analysis for rotational spectroscopy
=============================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   faq
   pyspectools.spectra

.. image:: _static/pst_logo_landscape.png


Introduction
------------
`PySpecTools` is a library written in Python 3.7 (and above) to help with
analyzing spectroscopy and quantum chemistry data.  The motivation behind this
is to provide an object-oriented way to analyze this kind of data, with
reproducibility, collaboration, and automation heavily in mind throughout the
development process - this makes the library flexible for both beginner users
(i.e. with little knowledge in Python) and for power users (those who are
proficient in Python).

As such, there are no explicitly programmed user interfaces for `PySpecTools` -
the tools were built with analysis using notebook environments such as
`Jupyter`_ notebook/lab, and removes most of the need for any custom
interfacing.

Key Features
~~~~~~~~~~~~

A list of common functionality can be found in the `FAQ`_ section, which details
some of the tasks you may wish to carry out but not sure how.

If there is a feature/analysis you would like to do but don't know how, please
raise an issue on Github!

API Reference
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 4

   modules 

.. _Jupyter: https://jupyter.org
.. _FAQ: faq


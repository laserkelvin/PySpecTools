import re
import os
import shutil
import stat
import platform
import sys
from warnings import warn
from distutils.command.sdist import sdist as _sdist
from distutils.extension import Extension
from distutils.spawn import find_executable
from glob import glob

import numpy as np
from setuptools import setup, find_packages
from setuptools.command.install import install

VERSIONFILE="pyspectools/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="pyspectools",
    description="A set of Python tools/routines for spectroscopy",
    author="Kelvin Lee",
    version=verstr,
    packages=find_packages(),
    include_package_data=True,
    author_email="kin.long.kelvin.lee@gmail.com",
    install_requires=[
        "numpy>=1.16",
        "pandas",
        "scipy",
        "matplotlib",
        "astroquery==0.3.8",
        "astropy==3.0.5",
        "lmfit",
        "peakutils>=1.3.2",
        "sklearn",
        "colorlover",
        "monsterurl",
        "plotly>=3.0.0",
        "periodictable",
        "uncertainties",
        "joblib",
        "ruamel.yaml",
        "paramiko",
        "jinja2",
        "tqdm",
        "tinydb",
        "networkx",
        "monsterurl",
        "torch"
    ],
)

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

class PostInstallCommand(install):
    """
    This class defines the functions to be run after running pip install.
    The routines to run are:
    1. Checking if SPFIT/SPCAT are installed
    """

    def check_pickett(self):
        for executable in ["spcat", "spfit", "calbak"]:
            if find_executable(executable) is None:
                warn(executable + " not found in PATH.")

    def setup_folders(self):
        """
        Sets up the dot folders that are utilized by the routines. If the matplotlib
        user folder doesn't exist, this function will make one.
        """
        folders = [
            ".pyspectools",
            ".pyspectools/templates",
            ".config/matplotlib/stylelib"
        ]
        folders = [
            os.path.join(os.path.expanduser("~"), folder) for folder in folders
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def setup_files(self):
        try:
            # Copy over matplotlib stylesheets
            home_dir = os.path.expanduser("~")
            if os.path.exists(
                home_dir + "/.config/matplotlib/stylelib"
            ) is False:
                os.mkdir(home_dir + "/.config/matplotlib/stylelib")
            for sheet in os.listdir("./pyspectools/mpl_stylesheets"):
                if ".yml" not in sheet:
                    path = os.path.expanduser("~") + \
                           "/.config/matplotlib/stylelib" + sheet
                else:
                    path = os.path.expanduser("~") + \
                        "/.pyspectools/" + sheet
                shutil.copy(sheet, path)
        except FileExistsError:
            pass

    def run(self):
        # Check for SPFIT/SPCAT executables in PATH
        self.check_pickett()
        install.run(self)

cmdclass = dict()

cmdclass.update(
    **{
        "develop": PostInstallCommand,
        "install": PostInstallCommand
    }
)

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
    cmdclass=cmdclass,
)

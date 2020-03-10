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

"""
    This recipe for including Cython in the setup.py was shamelessly
    taken from a StackOverflow answer:
    https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python
    -package-that-contains-cython-code
"""


class sdist(_sdist):
    """
    This class simply ensures that the latest .pyx modules are compiled into C when Cython is
    available. This way you don't have to manually compile the .pyx into C prior to pushing
    to git.
    """

    def run(self):
        from Cython.Build import cythonize
        _ = [cythonize(module) for module in glob("pyspectools/fast/*.pyx")]
        _sdist.run(self)


try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    use_cython = True
    print("Building Cython routines.")
except ImportError:
    print("Not building cython!")
    use_cython = False

# Determine if the system is Linux, which makes Cython
# require additional library args
if platform.system() == "Linux":
    libraries = ["m"]
else:
    libraries = list()


cmdclass = dict()
ext_modules = list()

if use_cython:
    ext_modules += [
        Extension(
            "pyspectools.fast.lineshapes",
            ["pyspectools/fast/lineshapes.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-ffast-math", "-march=native", "-O3"],
            libraries=libraries
        ),
        Extension(
            "pyspectools.fast.filters",
            ["pyspectools/fast/filters.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-ffast-math", "-march=native", "-O3"],
            libraries=libraries
        ),
        Extension(
            "pyspectools.fast.routines",
            ["pyspectools/fast/routines.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-ffast-math", "-march=native", "-O3"],
            libraries=libraries
        )
    ]
    cmdclass.update(**{"build_ext": build_ext, "sdist": sdist})
else:
    # If Cython is not available, then use the latest C files
    ext_modules += [
        Extension(
            "pyspectools.fast.lineshapes",
            ["pyspectools/fast/lineshapes.c"]
        ),
        Extension(
            "pyspectools.fast.filters",
            ["pyspectools/fast/filters.c"]
        ),
        Extension(
            "pyspectools.fast.routines",
            ["pyspectools/fast/routines.c"]
        )
    ]


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
            # Copy over YAML file containing the parameter coding
            # shutil.copy(
            #     "./pyspectools/pickett_terms.yml",
            #     os.path.expanduser("~") + "/.pyspectools/pickett_terms.yml"
            # )
            # # Copy over templates for molecule types
            # shutil.copytree(
            #     "./pyspectools/templates",
            #     os.path.expanduser("~") + "/.pyspectools/templates"
            # )
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

    def setup_scripts(self):
        # Set up the scripts and make them executable.
        # There's some hacking involved here because we need to grab
        # the anaconda python path to make the script know which
        # interpreter to use.
        format_dict = {"python_path": sys.executable}

        templates = glob("./scripts/*")
        if len(templates) == 0:
            pass
        else:
            for template in templates:
                template_name = template.split("/")[-1]
                with open(template, "r") as read_file:
                    file_contents = read_file.read()
                with open(os.path.expanduser(
                        "~") + "/.pyspectools/" + template_name,
                          "w+") as write_file:
                    write_file.write(file_contents.format(**format_dict))
                os.chmod(
                    os.path.expanduser("~") + "/.pyspectools/" + template_name,
                    stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)

    def run(self):
        # Check for SPFIT/SPCAT executables in PATH
        self.check_pickett()
        # Ensure folders exist
        # self.setup_folders()
        # Copy files for schemes and parameters over
        # self.setup_files()
        # Set up any scripts
        # self.setup_scripts()
        install.run(self)


cmdclass.update(
    **{
        "develop": PostInstallCommand,
        "install": PostInstallCommand
    }
)

setup(
    name="pyspectools",
    version="4.3.3",
    description="A set of Python tools/routines for spectroscopy",
    author="Kelvin Lee",
    packages=find_packages(),
    include_package_data=True,
    author_email="kin_long_kelvin.lee@cfa.harvard.edu",
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
        "monsterurl"
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules
)

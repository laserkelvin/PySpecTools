from setuptools import setup, find_packages
from setuptools.command.install import install 
from glob import glob
import os
from distutils.spawn import find_executable
import stat
import sys
import shutil

class PostInstallCommand(install):
    def check_pickett(self):
        for executable in ["spcat", "spfit", "calbak"]:
            if find_executable(executable) is None:
                print(executable + " not found in PATH.")
                print("Make sure SPFIT/SPCAT is in your path.")

    def setup_files(self):
        # If the dotfolder is not present in home directory, make one
        if os.path.isdir(os.path.expanduser("~") + "/.pyspectools") is False:
            os.mkdir(os.path.expanduser("~") + "/.pyspectools")

        try:
            # Copy over YAML file containing the parameter coding
            shutil.copy2(
                "./pyspectools/pickett_terms.yml",
                os.path.expanduser("~") + "/.pyspectools/pickett_terms.yml"
            )
            # Copy over templates for molecule types
            shutil.copytree(
                "./pyspectools/templates",
                os.path.expanduser("~") + "/.pyspectools/templates"
            )
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
                with open(os.path.expanduser("~") + "/.pyspectools/" + template_name, "w+") as write_file:
                    write_file.write(file_contents.format(**format_dict))
                os.chmod(os.path.expanduser("~") + "/.pyspectools/" + template_name,
                         stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)

    def run(self):
        # Check for SPFIT/SPCAT executables in PATH
        self.check_pickett()
        # Copy files for schemes and parameters over
        self.setup_files()
        # Set up any scripts
        self.setup_scripts()
        install.run(self)

setup(
    name="pyspectools",
    version="3.2.0",
    description="A set of Python tools/routines for spectroscopy",
    author="Kelvin Lee",
    packages=find_packages(),
    include_package_data=True,
    author_email="kin_long_kelvin.lee@cfa.harvard.edu",
    install_requires=[
        "numpy>=1.15",
        "pandas",
        "scipy",
        "matplotlib",
        "astroquery",
        "astropy",
        "lmfit",
        "peakutils",
        "sklearn",
        "colorlover",
        "plotly>=3.0.0",
        "periodictable",
        "uncertainties",
        "joblib",
        "ruamel.yaml",
        "paramiko",
        "jinja2"
    ],
    cmdclass={
        "develop": PostInstallCommand,
        "install": PostInstallCommand
    }
)

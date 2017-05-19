from setuptools import setup
from setuptools.command.install import install
from glob import glob
import os
from distutils.spawn import find_executable
import stat
import sys

class PostInstallCommand(install):
    def check_pickett(self):
        for executable in ["spcat", "spfit", "calbak"]:
            if find_executable(executable) is None:
                print(executable + " not found in PATH.")
                print("Make sure SPFIT/SPCAT is in your path.")

    def run(self):
        self.check_pickett()
        format_dict = {"python_path": sys.executable}
        if os.path.isdir(os.path.expanduser("~") + "/bin") is False:
            os.mkdir(os.path.expanduser("~") + "/bin")

        templates = glob("./scripts/*")
        if len(templates) == 0:
            pass
        else:
            for template in templates:
                template_name = template.split("/")[-1]
                with open(template, "r") as read_file:
                    file_contents = read_file.read()
                with open(os.path.expanduser("~") + "/bin/" + template_name, "w+") as write_file:
                    write_file.write(file_contents.format(**format_dict))
                os.chmod(os.path.expanduser("~") + "/bin/" + template_name,
                         stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
        install.run(self)

setup(
    name="pyspectools",
    version="0.3.1",
    description="A set of Python tools/routines for spectroscopy",
    author="Kelvin Lee",
    packages=["pyspectools"],
    include_package_data=True,
    author_email="kin_long_kelvin.lee@cfa.harvard.edu",
    install_requires=[
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "peakutils"
    ],
    cmdclass={
        "develop": PostInstallCommand,
        "install": PostInstallCommand
    }
)

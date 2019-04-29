""" Routines to:
    Parse cat files
    Run SPFIT and/or SPCAT
"""

import os
import subprocess
import shutil
import json
import types
from glob import glob

import ruamel.yaml as yaml
import numpy as np
import joblib
import paramiko

from pyspectools import pypickett as pp


def run_spcat(filename, temperature=None):
    # Run SPCAT
    parameter_file = filename + ".var"
    if os.path.isfile(filename + ".var") is False:
        print("VAR file unavailable. Attempting to run with PAR file.")
        if os.path.isfile(filename + ".par") is False:
            raise FileNotFoundError("No .var or .par file found.")
        else:
            shutil.copy2(filename + ".par", parameter_file)
    process = subprocess.Popen(["spcat", filename + ".int", parameter_file],
                     stdout=subprocess.PIPE            # suppress stdout
                     )
    process.wait()
    # Extract the partition function at the specified temperature
    if temperature is not None:
        # Read in the piped standard output, and format into a list
        stdout = str(process.communicate()[0]).split("\\n")
        for line in stdout:
            if temperature in line:
                # If the specified temperature is found, get the partition
                # function
                Q = float(line.split()[1])
        return Q


def run_calbak(filename):
    """ Runs the calbak routine, which generates a .lin file from the .cat """
    if os.path.isfile(filename + ".cat") is False:
        raise FileNotFoundError(filename + ".cat is missing; cannot run calbak.")
    process = subprocess.Popen(
        [
            "calbak",
            filename + ".cat",
            filename + ".lin"
        ],
        stdout=subprocess.DEVNULL
    )
    process.wait()
    with open(filename + ".lin") as read_file:
        lin_length = read_file.readlines()
    if lin_length == 0:
        raise RuntimeError("No lines produced in calbak! Check .cat file.")


def run_spfit(filename):
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    process = subprocess.run(
        ["spfit", filename + ".lin", filename + ".par"],
        timeout=20.
        )
    if process.returncode != 0:
        raise OSError("SPFIT failed to run.")


def pickett_molecule(json_filepath=None):
    # Provide a JSON file with all the Pickett settings, and generate an
    # instance of the molecule class
    # This method is superceded by serializing using classmethods for each
    # file format
    raise UserWarning("pickett_molecule is now outdated. Please use the class \
                       methods from_yaml or from_json.")
    if json_filepath is None:
        print("No JSON input file specified.")
        print("A template file will be created in your directory; please rerun\
               after setting up the parameters.")
        copy_template()
        raise FileNotFoundError("No input file specified.")
    json_data = read_json(json_filepath)
    molecule_object = pp.MoleculeFit(json_data)
    return molecule_object


def human2pickett(name, reduction="A", linear=True, nuclei=0):
    """ Function for translating a Hamiltonian parameter to a Pickett
        identifier.

        An alternative way of doing this is to programmatically
        generate the Pickett identifiers, and just use format string
        to output the identifier.
    """
    pickett_parameters = read_yaml(
        os.path.expanduser("~") + "/.pyspectools/pickett_terms.yml"
    )
    if name is "B" and linear is True:
        # Haven't thought of a clever way of doing this yet...
        identifier = 100
    elif name is "B" and linear is False:
        identifier = 20000
    else:
        # Hyperfine terms
        if name in ["eQq", "eQq/2"]:
            identifier = str(pickett_parameters[name]).format(nuclei)
        elif "D_" in name or "del" in name:
            identifier = str(pickett_parameters[name][reduction])
        else:
            try:
                identifier = pickett_parameters[name]
            except KeyError:
                print("Parameter name unknown!")
    return identifier


def read_json(json_filepath):
    with open(json_filepath, "r") as read_file:
        json_data = json.load(read_file)
    return json_data


def dump_json(json_filepath, json_dict):
    """ Function for dumping a Python dictionary to JSON syntax.
        Does so with some pretty formatting with indents and whatnot.
    """
    with open(json_filepath, "w+") as write_file:
        json.dump(json_dict, write_file, indent=4, sort_keys=True)


def read_yaml(yaml_filepath):
    """ Function for reading a YAML file in as a Python dictionary """
    with open(yaml_filepath) as read_file:
        yaml_data = yaml.load(read_file, Loader=yaml.Loader)
    return yaml_data


def dump_yaml(yaml_filepath, yaml_dict):
    """ Function for dumping a python dictionary to
        YAML syntax
    """
    with open(yaml_filepath, "w+") as write_file:
        yaml.dump(yaml_dict, write_file)


def generate_folder():
    """
    Generates the folder for the next calculation
    and returns the next calculation number
    """
    folderlist = list_directories()      # get every file/folder in directory
    # filter out any non-folders that happen to be here
    shortlist = list()
    for folder in folderlist:
        try:
            shortlist.append(int(folder))
        except ValueError:                  # if it's not an integer
            pass
    if len(shortlist) == 0:
        lastcalc = 0
    else:
        lastcalc = max(shortlist)
    #lastcalc = len(folderlist)
    os.mkdir(str(lastcalc + 1))
    return lastcalc + 1


def format_uncertainty(value, uncertainty):
    """ Function to determine the number of decimal places to
        format the uncertainty. Probably not the most elegant way of doing this.
    """
    # Convert the value into a string, then determine the length by
    # splitting at the decimal point
    decimal_places = decimal_length(value)
    uncertainty = float(uncertainty)           # make sure we're dealing floats
    uncertainty_places = decimal_length(uncertainty)
    # Force the uncertainty into decimals
    uncertainty = uncertainty * 10**-uncertainty_places[1]
    # Work out how many places we've moved now
    uncertainty_places = decimal_length(uncertainty)
    # Move the precision of the uncertainty to match the precision of the value
    uncertainty = uncertainty * 10**(uncertainty_places[1] - decimal_places[1])
    return uncertainty


def decimal_length(value):
    # Function that determines the decimal length of a float; convert the value
    # into a string, then work out the length by splitting at the decimal point
    decimal_split = str(value).split(".")
    return [len(position) for position in decimal_split]


def copy_template():
    script_location = os.path.dirname(os.path.realpath(__file__))
    templates_folder = script_location + "/templates/"
    available_templates = glob(templates_folder + "*.json")
    available_templates = [template.split("/")[-1] for template in available_templates]
    print("The templates available are:")
    for template in available_templates:
        print(template)
    target = input("Please specify which template to copy:      ")
    if target not in available_templates:
        print("Not a template; probably a typo.")
        print("Please re-run the script.")
    else:
        shutil.copy2(templates_folder + target, os.getcwd() + "/parameters.json")
        print("Copied template " + target + " to your folder as parameters.json.")
        print("Edit the .json input file and re-run the script.")


def list_directories():
    return [directory for directory in os.listdir() if os.path.isdir(directory)]


def backup_files(molecule_name, save_location):
    extensions = [".cat", ".var", ".par", ".int", ".json", ".lin"]
    filenames = [molecule_name + ext for ext in extensions]
    for filename in filenames:
        if os.path.isfile(filename) is True:
            shutil.copy2(filename, save_location)
            print("Backing up " + filename + " to " + save_location)
        else:
            pass


def isnotebook():
    # Check if the code is being run in a notebook, IPython shell, or Python
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def save_obj(obj, filepath, **kwargs):
    """
        Function to serialize an object using dump from joblib.

        Additional kwargs are passed into the dump, which can
        be compression parameters, etc.

        parameters:
        ---------------
        obj - instance of object to be serialized
        filepath - filepath to save to
    """
    settings = {
        "compress": ("gzip", 6),
        "protocol": 4
        }
    settings.update(kwargs)
    joblib.dump(obj, filepath, **settings)


def read_obj(filepath):
    """
        Wrapper for joblib.load to load an object from disk

        parameters:
        ---------------
        filepath - path to object
    """
    obj = joblib.load(filepath)
    return obj


def dump_packages():
    """
        Function that will return a list of packages that
        have been loaded and their version numbers.

        This function will ignore system packages:
        sys, __builtins__, types, os
        
        as well as modules with no version.


        This is not working the way I want it to...

        returns:
        -------------
        mod_dict - dict with keys corresponding to module name,
                   and values the version number.
    """
    mod_dict = dict()
    sys_packages = ["sys", "__builtins__", "types", "os"]
    for name, module in globals().items():
        if isinstance(module, types.ModuleType):
            if module.__name__ not in sys_packages:
                try:
                    mod_name = module.__name__
                    mod_ver = module.__version__
                    mod_dict[mod_name] = mod_ver
                except AttributeError:
                    pass
    return mod_dict


def find_nearest(array, value):
    """
    Search a numpy array for the closest value.
    :param array: np.array of values
    :param value: float value to search array for
    :return: value of array closest to target, and the index
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


class RemoteClient(paramiko.SSHClient):
    def __init__(self, hostname=None, username=None, **kwargs):
        super().__init__()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connect(
            hostname=hostname,
            username=username,
            **kwargs
        )

        self.sftp = self.open_sftp()

    @classmethod
    def from_file(cls, filepath):
        """
        Reload a remote session from a pickle file created by the save_session.
        :param filepath: str path to RemoteClient pickle file
        :return: RemoteClient object
        """
        remote = read_obj(filepath)
        # Make sure that the pickle file is a RemoteClient object
        if remote.__name__ != "RemoteClient":
            raise Exception("File was not a RemoteClient session; {}".format(remote.__name__))
        else:
            return read_obj(filepath)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Dunder method that should be called when the object is destroyed. In this case,
        the remote connection should be closed automatically.
        """
        self.sftp.close()
        self.close()

    def get_file(self, remote_path, local_path=os.getcwd()):
        """
        Download a file from remote server to disk. If no local path is provided, defaults
        to the current working directory.
        :param remote_path: str remote file path target
        :param local_path: str optional path to save the file to
        """
        self.sftp.get(remote_path, local_path)

    def run_command(self, command):
        stdin, stdout, stderr = self.exec_command(command)
        error_msg = stderr.read()
        if len(error_msg) == 0:
            return stdout.readlines()
        else:
            raise Exception("Error in running command: {}".format(error_msg))

    def open_remote(self, remote_path):
        """
        Function to stream the file contents of a remote file. Can be used to directly
        provide data into memory without downloading it to disk.
        :param remote_path: str remote path to target file
        :return: list of contents of the target file
        """
        contents = self.run_command("cat {}".format(remote_path))
        return contents

    def ls(self, remote_path=""):
        """
        Function to get the list of files present in a specified directory.
        Defaults to the current ssh directory.
        :param remote_path: str remote path to inspect
        :return: list of files and folders
        """
        contents = self.run_command("ls {}".format(remote_path))
        return contents

    def save_session(self, filepath="ssh.pkl", **kwargs):
        """
        Function to dump the ssh settings object to a pickle file. Keep in mind
        that while this is a matter of convenience, the file is unencrypted and
        so storing passwords in here is not exactly the safest thing to do!
        :param filepath: str optional path to save the session to.
        """
        save_obj(self, filepath, **kwargs)


def search_file(root_path, filename, ext=".txt"):
    """
    Function for searching for a specific filename in a given root path.
    The option `ext` specifies whether or not the extension must also be matched.
    :param root: str path to begin search
    :param filename: str filename to match
    :param ext: bool option to match file extension also
    :return: str full path to the file
    """
    for root, dirs, files in os.walk(root_path):
        if any([file for file in files if filename in file]) is True:
            return os.path.join(root, filename) + ext


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

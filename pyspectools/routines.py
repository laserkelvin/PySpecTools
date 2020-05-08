""" Routines to:
    Parse cat files
    Run SPFIT and/or SPCAT
"""

import os
import subprocess
import shutil
import json
import types
from typing import List, Any, Union, Dict, Tuple
from glob import glob
from warnings import warn

import ruamel.yaml as yaml
import numpy as np
import joblib
import paramiko


def run_spcat(filename: str, temperature=None):
    # Run SPCAT
    parameter_file = filename + ".var"
    if os.path.isfile(filename + ".var") is False:
        print("VAR file unavailable. Attempting to run with PAR file.")
        if os.path.isfile(filename + ".par") is False:
            raise FileNotFoundError("No .var or .par file found.")
        else:
            shutil.copy2(filename + ".par", parameter_file)
    process = subprocess.Popen(
        ["spcat", filename + ".int", parameter_file],
        stdout=subprocess.PIPE,  # suppress stdout
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


def run_calbak(filename: str):
    """ Runs the calbak routine, which generates a .lin file from the .cat """
    if os.path.isfile(filename + ".cat") is False:
        raise FileNotFoundError(filename + ".cat is missing; cannot run calbak.")
    process = subprocess.Popen(
        ["calbak", filename + ".cat", filename + ".lin"], stdout=subprocess.DEVNULL
    )
    process.wait()
    with open(filename + ".lin") as read_file:
        lin_length = read_file.readlines()
    if lin_length == 0:
        raise RuntimeError("No lines produced in calbak! Check .cat file.")


def run_spfit(filename: str):
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    process = subprocess.run(
        ["spfit", filename + ".lin", filename + ".par"],
        timeout=20.0,
        capture_output=True,
    )
    if process.returncode != 0:
        raise OSError("SPFIT failed to run.")


def list_chunks(target: List[Any], n: int):
    """
    Split a list into a number of chunks with length n. If there are not enough elements,
    the last chunk will finish the remaining elements.

    Parameters
    ----------
    target: list
        List to split into chunks
    n: int
        Number of elements per chunk

    Returns
    -------
    split_list: list
        Nested list of chunks
    """
    split_list = [target[i : i + n] for i in range(0, len(target), n)]
    return split_list


def human2pickett(name: str, reduction="A", linear=True, nuclei=0):
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


def read_json(json_filepath: str) -> Dict[Any, Any]:
    """
    Load a JSON file into memory as a Python dictionary.
    
    Parameters
    ----------
    json_filepath : str
        Path to the JSON file
    
    Returns
    -------
    Dict[Any, Any]
        Dictionary from JSON file
    """
    with open(json_filepath, "r") as read_file:
        json_data = json.load(read_file)
    return json_data


def dump_json(json_filepath: str, json_dict: Dict[Any, Any]):
    """
    Function to serialize a Python dictionary into a JSON file.
    The pretty printing is enabled by default.
    
    Parameters
    ----------
    json_filepath : str
        Path to the JSON file to save to
    json_dict : Dict[Any, Any]
        Dictionary to be serialized
    """
    with open(json_filepath, "w+") as write_file:
        json.dump(json_dict, write_file, indent=4, sort_keys=True)


def read_yaml(yaml_filepath: str) -> Dict[Any, Any]:
    """
    Function to load in a YAML file into a Python dictionary.
    
    Parameters
    ----------
    yaml_filepath : str
        Path to the YAML file
    
    Returns
    -------
    Dict[Any, Any]
        Dictionary based on the YAML contents
    """
    with open(yaml_filepath) as read_file:
        yaml_data = yaml.load(read_file, Loader=yaml.Loader)
    return yaml_data


def dump_yaml(yaml_filepath: str, yaml_dict: Dict[Any, Any]):
    """
    Function to serialize a Python dictionary into a YAML file.
    
    Parameters
    ----------
    yaml_filepath : str
        Path to the YAML file
    yaml_dict : Dict[Any, Any]
        Dictionary to be serialized
    """
    with open(yaml_filepath, "w+") as write_file:
        yaml.dump(yaml_dict, write_file)


def generate_folder():
    """
    Generates the folder for the next calculation
    and returns the next calculation number
    """
    folderlist = list_directories()  # get every file/folder in directory
    # filter out any non-folders that happen to be here
    shortlist = list()
    for folder in folderlist:
        try:
            shortlist.append(int(folder))
        except ValueError:  # if it's not an integer
            pass
    if len(shortlist) == 0:
        lastcalc = 0
    else:
        lastcalc = max(shortlist)
    # lastcalc = len(folderlist)
    os.mkdir(str(lastcalc + 1))
    return lastcalc + 1


def format_uncertainty(value: float, uncertainty: float):
    """ Function to determine the number of decimal places to
        format the uncertainty. Probably not the most elegant way of doing this.
    """
    # Convert the value into a string, then determine the length by
    # splitting at the decimal point
    decimal_places = decimal_length(value)
    uncertainty = float(uncertainty)  # make sure we're dealing floats
    uncertainty_places = decimal_length(uncertainty)
    # Force the uncertainty into decimals
    uncertainty = uncertainty * 10 ** -uncertainty_places[1]
    # Work out how many places we've moved now
    uncertainty_places = decimal_length(uncertainty)
    # Move the precision of the uncertainty to match the precision of the value
    uncertainty = uncertainty * 10 ** (uncertainty_places[1] - decimal_places[1])
    return uncertainty


def decimal_length(value: float):
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


def flatten_list(input_list: List[List[Any]]):
    """
    Takes a nested list of values and flattens it. The code is written as a try/except that makes the assumption
    that the data is a list/tuple/array, and in the case that it isn't will simply append the item to the
    output instead.

    Parameters
    ----------
    input_list: list
        List of values, where some of the elements are lists

    Returns
    -------
    output_list: list
        Flattened version of input_list
    """
    output_list = list()
    for value in input_list:
        try:
            output_list.extend(value)
        # Ask for forgiveness
        except TypeError:
            output_list.append(value)
    return output_list


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
        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole?
            return True
        elif shell == "TerminalInteractiveShell":  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def save_obj(obj: Any, filepath: str, **kwargs):
    """
        Function to serialize an object using dump from joblib.

        Additional kwargs are passed into the dump, which can
        be compression parameters, etc.

        parameters:
        ---------------
        obj - instance of object to be serialized
        filepath - filepath to save to
    """
    settings = {"compress": ("gzip", 6), "protocol": 4}
    settings.update(kwargs)
    joblib.dump(obj, filepath, **settings)


def read_obj(filepath: str):
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


def find_nearest(array: np.ndarray, value: Union[float, int]) -> Tuple[np.ndarray, int]:
    """
    Function that will find the nearest value in a NumPy array to a specified
    value.
    
    Parameters
    ----------
    array : np.ndarray
        NumPy 1D array
    value : float
        Value to search the array for
    
    Returns
    -------
    Tuple[np.ndarray, int]
        Returns the closest value, as well as the index
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


class RemoteClient(paramiko.SSHClient):
    def __init__(self, hostname=None, username=None, **kwargs):
        super().__init__()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connect(hostname=hostname, username=username, **kwargs)

        self.sftp = self.open_sftp()

    @classmethod
    def from_file(cls, filepath: str):
        """
        Reload a remote session from a pickle file created by the save_session.
        :param filepath: str path to RemoteClient pickle file
        :return: RemoteClient object
        """
        remote = read_obj(filepath)
        # Make sure that the pickle file is a RemoteClient object
        if remote.__name__ != "RemoteClient":
            raise Exception(
                "File was not a RemoteClient session; {}".format(remote.__name__)
            )
        else:
            return read_obj(filepath)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Dunder method that should be called when the object is destroyed. In this case,
        the remote connection should be closed automatically.
        """
        self.sftp.close()
        self.close()

    def get_file(self, remote_path: str, local_path=os.getcwd()):
        """
        Download a file from remote server to disk. If no local path is provided, defaults
        to the current working directory.
        :param remote_path: str remote file path target
        :param local_path: str optional path to save the file to
        """
        self.sftp.get(remote_path, local_path)

    def run_command(self, command: str):
        stdin, stdout, stderr = self.exec_command(command)
        error_msg = stderr.read()
        if len(error_msg) == 0:
            return stdout.readlines()
        else:
            raise Exception(f"Error in running command: {error_msg}")

    def open_remote(self, remote_path: str):
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


def group_consecutives(vals: List[float], step=1):
    """
    Function to group all consecutive values in a list together. The primary purpose of this
    is to split concatenated spectra that are given in a single list of frequencies
    into individual windows.
    
    Parameters
    ----------
    vals : list
        List of floats to be split
    step : int, optional
        [description], by default 1
    
    Returns
    -------
    [type]
        [description]
    """
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

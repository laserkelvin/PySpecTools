import contextlib
import os
import re
from pathlib import Path
from subprocess import run, PIPE
from tempfile import TemporaryDirectory
from typing import Union, Dict, Number, Type

import numpy as np

from pyspectools import routines
from pyspectools.pypickett.classes import AsymmetricTop, SymmetricTop, LinearMolecule, AbstractMolecule


par_template = """PySpecTools SPCAT input
 100  255    1    0    0.0000E+000    1.0000E+003    1.0000E+000 1.0000000000
{reduction}   {quanta}    {top}    0   {k_max}    0    {weight_axis}    {even_weight}    {odd_weight}     0   1   0
{parameters}
"""

int_template = """PySpecTools SPCAT input
 0  {mol_id}   {q:.4f}   0   {max_f_qno}  {int_min:.1f}  {int_max:.1f}   {freq_limit:.4f}  {T:.2f}
{dipole_moments}
"""


@contextlib.contextmanager
def work_in_temp():
    """
    Context manager for working in a temporary directory. This
    simply uses the `tempfile.TemporaryDirectory` to create and
    destroy the folder after using, and manages moving in and
    out of the folder.
    """
    cwd = os.getcwd()
    try:
        with TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            yield
    finally:
        os.chdir(cwd)


def run_spcat(filename: str, read_Q: bool = False, debug: bool = False):
    """
    Run SPCAT, and optionally parse the partition functions.

    Parameters
    ----------
    filename : str
        Name of the SPCAT file, without file extensions
    read_Q : bool, optional
        [description], by default False

    Returns
    -------
    List[float, np.ndarray]
        Returns the value used for Q, and if `read_Q` then
        a 2D NumPy array containing the temperature, Q, and log Q.
        Otherwise, `q_array` is returned as `None`.
    """
    # get rid of the stuff at the end
    filename = Path(filename)
    if filename.suffix != "":
        filename = filename.stem
    else:
        filename = str(filename)
    proc = run(["spcat", filename], stdout=PIPE)
    if debug:
        for ext in [".var", ".int"]:
            with open(f"{filename}{ext}", "r") as read_file:
                print("".join(read_file.readlines()[:10]))
    spcat_out = proc.stdout.decode("utf-8")
    if debug and Path(f"{filename}.cat").exists():
        with open(f"{filename}.cat") as read_file:
            print("".join(read_file.readlines()[:10]))
    q_array = list() if read_Q else None
    # get the initial Q value used
    for line in spcat_out.split("\n"):
        if "INITIAL Q" in line:
            line = line.replace(",", " ")
            initial_q = float(line.split()[3])
            # if we're not reading the partition function,
            # we stop here
            if not read_Q:
                break
        else:
            try:
                # for lines that actually have Q(T)
                if line.strip()[0].isdigit():
                    # t, q, log_q
                    values = [float(value) for value in line.split()]
                    q_array.append(values)
            except IndexError:
                pass
    if q_array is not None:
        q_array = np.vstack(q_array).T
    return (initial_q, q_array)


def write_qpart_file(filepath: Union[str, Path], q_array: np.ndarray):
    """
    Write the partition function out in the `molsim` format.

    Parameters
    ----------
    filepath : str
        [description]
    q_array : np.ndarray
        [description]
    """
    # if the filepath is a string, convert to a Path object
    if isinstance(filepath, str):
        filepath = Path(filepath)
    # if there's no file extension, add .qpart
    if filepath.suffix == "":
        filepath = filepath.with_suffix(".qpart")
    with open(filepath, "w+") as write_file:
        write_file.write("# form : interpolation\n")
        for row in q_array:
            write_file.write(f"{row[0]:.1f} {row[1]:.4f}\n")


def sanitize_keys(data: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
    new_data = data.copy()
    for key, value in data.items():
        if "chi" in key:
            # for times where the digit is omitted, tack it on
            if not key[-1].isdigit():
                new_key = f"{key}_1"
            else:
                new_key = key
            # if this is a diagonal element, we need to multiply
            # by 3/2 because SPCAT
            if re.match(r"chi_(\w)\1_\d", new_key):
                value *= 1.5
            new_data[new_key] = value
            if new_key != key:
                del new_data[key]
        # negate the value because SPCAT CD terms are negative
        elif "delta_" in key.lower() or "D_" in key:
            new_data[key] = -value
        # standardize mu_ with u_
        elif "mu_" in key:
            axis = key.split("_")[-1]
            new_data[f"u_{axis}"] = value
            del new_data[key]
        else:
            new_data[key] = value
    return new_data


def infer_molecule(parameters: Dict[str, Union[str, Number]]) -> Type[AbstractMolecule]:
    """
    Infers what rotor type to use based on what parameters
    have been provided. If the A, B, C rotational constant is present,
    we're looking at an asymmetric top. If only A and B are given,
    then it's a symmetric top. Finally, anything else is treated as
    a linear molecule.

    Parameters
    ----------
    parameters : Dict[str, Union[str, Number]]
        Rotational parameters for a molecule

    Returns
    -------
    Type[AbstractMolecule]
        Class reference to the rotor type
    """
    if all([key in parameters.keys() for key in ["A", "B", "C"]]):
        return AsymmetricTop
    elif all([key in parameters.keys() for key in ["A", "B"]]):
        return SymmetricTop
    else:
        return LinearMolecule


def load_molecule_yaml(filepath: Union[str, Path]):
    """
    Parses a YAML file that contains standardized molecule
    parameter specifications, as well as associated metadata.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the YAML file
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    data = routines.read_yaml(filepath)
    # standardize the key/values for SPCAT
    data = sanitize_keys(data)
    hash = routines.hash_file(filepath)
    meta_keys = ["name", "doi", "notes", "smiles", "formula", "author"]
    metadata = {"md5": hash}
    # extract out the metadata
    for key in meta_keys:
        value = data.get(key)
        if value:
            metadata[key] = value
        # the data dict is exclusively for molecular parameters
        del data[key]
    # now we pick which molecule to use
    mol_type = infer_molecule(data)
    molecule = mol_type(**data)
    return (molecule, metadata)

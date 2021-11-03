import re
from shutil import copy2
from abc import ABC, abstractproperty
from typing import Dict, List, Union, Type, Tuple
from warnings import warn
from functools import wraps
from pathlib import Path
from typing import List, Dict
from difflib import get_close_matches
from glob import glob

import numpy as np

from pyspectools import routines
from pyspectools.units import kappa
from pyspectools.pypickett.utils import (
    par_template,
    int_template,
    work_in_temp,
    run_spcat,
    sanitize_keys,
)


def hyperfine_nuclei(method):
    """
    Defines a decorator that dynamically generates hyperfine
    nuclei coding, which returns the coding mapping based on
    what the user provides with respect to "chi_xx" parameters.

    The off-diagonal elements are hardcoded because they're
    not particularly programmatic.
    """

    @wraps(method)
    def reparameterize(molecule_obj):
        coding = method(molecule_obj)
        hyperfine_names = ["chi_aa", "chi_bb", "chi_cc"]
        # only operate when there are nuclei
        if molecule_obj.num_nuclei != 0:
            for nucleus in molecule_obj.nuclei:
                for index, name in enumerate(hyperfine_names):
                    hf_code = f"{nucleus}{nucleus}00{index+1}0000"
                    coding[f"{name}_{nucleus}"] = hf_code
                # add some extra ones
                coding[f"chi_bb-chi_cc_{nucleus}"] = f"{nucleus}{nucleus}0040000"
                coding[f"chi_ab_{nucleus}"] = f"{nucleus}{nucleus}0610000"
                coding[f"chi_bc_{nucleus}"] = f"{nucleus}{nucleus}0210000"
                coding[f"chi_ac_{nucleus}"] = f"{nucleus}{nucleus}0410000"
        # override with the user specified terms at the end
        coding.update(molecule_obj.custom_coding)
        return coding

    return reparameterize


class Parameter(object):
    def __init__(self, name: str, value: float, unc: float = 0.0):
        self.name = name
        self._value = value
        self._unc = unc

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    @property
    def unc(self) -> float:
        return self._unc

    @unc.setter
    def unc(self, value: float):
        self._unc = value

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4e}+/-{self.unc:.4e}"

    @classmethod
    def from_dict(cls, **kwargs):
        sanitized = {key.replace("_", ""): value for key, value in kwargs.items()}
        return cls(**sanitized)


class AbstractMolecule(ABC):
    def __init__(
        self,
        custom_coding: Union[None, Dict[str, Union[str, int]]] = None,
        spins: Union[None, List[int], Dict[int, int]] = None,
        **params,
    ):
        self._nuclei = dict()
        self._custom_coding = dict() if not custom_coding else custom_coding
        for key, value in params.items():
            try:
                # if the input is iterable, unpack
                param = Parameter(key, *value)
            except TypeError:
                # otherwise just set the value to single number
                param = Parameter(key, value)
            # check for hyperfine spins by looking for chi
            if "chi" in key.lower():
                number = int(key.split("_")[-1])
                # if no spins are provided, assume it's nitrogen
                if spins is None:
                    spin = 1
                # deal with the other cases
                elif isinstance(spins, list):
                    spin = spins[number - 1]
                elif isinstance(spins, dict):
                    spin = spins.get(number)
                if number not in self._nuclei:
                    self._nuclei[number] = spin
            setattr(self, key, param)
        self.param_names = list(params.keys())

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {name: param.__dict__ for name, param in self.params.items()}

    @property
    def params(self) -> Dict[str, Type[Parameter]]:
        params = {key: getattr(self, key) for key in self.param_names}
        return params

    @property
    def nuclei(self) -> Dict[str, int]:
        return self._nuclei

    @property
    def num_nuclei(self) -> int:
        return len(self._nuclei)

    @property
    def multiplicity(self) -> int:
        if len(self.nuclei) != 0:
            return "".join([str(2 * s + 1) for s in self.nuclei.values()])
        else:
            return 1

    @property
    def custom_coding(self) -> Dict[str, Union[str, int]]:
        return self._custom_coding

    @custom_coding.setter
    def custom_coding(self, value: Union[None, Dict[str, Union[str, int]]]):
        """
        Update the custom coding stored in the instance. If the
        user tries to set the value to `None`, the coding is reset
        to an empty dictionary. If a dictionary is passed, assuming
        keys are the Hamiltonian parameters and values are the SPCAT
        coding, then the coding dictionary is updated. For example:

        ```
        obj.custom_coding = {"B": 100}     # updates the coding
        obj.custom_coding = None           # resets the coding
        ```

        Parameters
        ----------
        value : Union[None, Dict[str, Union[str, int]]]
            If `None`, reset any custom coding prior. Otherwise,
            a dictionary will be used to update the coding.
        """
        if not value:
            self._custom_coding = dict()
        else:
            self._custom_coding.update(value)

    @abstractproperty
    def param_coding(self) -> Dict[str, Union[str, int]]:
        """
        Implement a dictionary mapping between string names of
        Hamiltonian parameters to Fourier coding, for a particular
        class of molecule

        Returns
        -------
        Dict[str]
            key corresponds to string name, value is the Fourier
            coding
        """
        pass

    def __repr__(self) -> str:
        combined = list()
        for key, value in self.params.items():
            coding = self.param_coding.get(key)
            # this ensures that only "real" parameters are exported
            if not coding:
                valid_keys = self.param_coding.keys()
                # get up to three of the closest matches
                close = get_close_matches(key, valid_keys, n=3)
                warn(
                    f"{key} has not been implemented in {self.__class__.__name__} and was ignored.\nClosest matches are {close}"
                )
            else:
                combined.append(
                    f"{coding:>14}  {value.value:>22e} {value.unc:>15e} /{key}"
                )
        return "\n".join(combined)

    def to_yaml(self, filepath: Union[str, Type[Path]]) -> None:
        output = {key: (value.value, value.unc) for key, value in self.params.items()}
        routines.dump_yaml(filepath, output)

    @classmethod
    def from_yml(cls, filepath: Union[str, Type[Path]]):
        data = routines.read_yaml(filepath)
        return cls(**data)

    @property
    def type(self) -> str:
        return self.__class__.__name__


class LinearMolecule(AbstractMolecule):
    def __init__(
        self,
        custom_coding: Union[None, Dict[str, Union[str, int]]] = None,
        spins: Union[None, List[int], Dict[int, int]] = None,
        **params,
    ):
        super().__init__(custom_coding, spins, **params)

    @property
    @hyperfine_nuclei
    def param_coding(self) -> Dict[str, Union[str, int]]:
        coding = {"B": 100, "D": 200, "H": 300, "L": 400}
        return coding


class SymmetricTop(LinearMolecule):
    def __init__(
        self,
        custom_coding: Union[None, Dict[str, Union[str, int]]] = None,
        spins: Union[None, List[int], Dict[int, int]] = None,
        **params,
    ):
        super().__init__(custom_coding, spins, **params)

    @property
    @hyperfine_nuclei
    def param_coding(self) -> Dict[str, Union[str, int]]:
        # inherit coding from the linear molecule
        coding = super().param_coding
        coding.update(
            {
                "A-B": 1000,
                # main differences are in the quartic terms
                "D_K": 2000,
                "D_JK": 1100,
                "D_J": coding.get("D"),  # this basically sets up an alias
                "Delta_J": coding.get("D"),
                "Delta_JK": 1100,
                "Delta_K": 2000,
                "del_J": 40100,
                "del_K": 41000,
                "d_1": 40100,
                "d_2": 50000,
                # sextic constants
                "H_K": 3000,
                "H_JK": 1200,
                "H_KJ": 2100,
                "H_J": coding.get("H"),
                "h_1": 40200,
                "h_2": 50100,
                "h_3": 60000,
                "Phi_J": coding.get("H"),
                "Phi_JK": 1200,
                "Phi_KJ": 2100,
                "Phi_K": 3000,
                "phi_J": 40200,
                "phi_JK": 41100,
                "phi_K": 42000,
                "L_J": coding.get("L"),  # octic terms
                "L_JJK": 1300,
                "L_JK": 2200,
                "L_KKJ": 3100,
                "L_K": 4000,
                "l_J": 40300,  # A-reduced, octic
                "l_JK": 41200,
                "l_KJ": 42100,
                "l_K": 43000,
                "l_1": 40300,  # S-reduced, octic
                "l_2": 50200,
                "l_3": 60100,
                "l_4": 70000,
                "P_J": 500,  # decadic terms
                "P_JJK": 1400,
                "P_JK": 2300,
                "P_KJ": 3200,
                "P_KKJ": 4100,
                "P_K": 5000,
                "p_J": 40400,  # A-reduced, decadic
                "p_JJK": 41300,
                "p_JK": 42200,
                "p_KKJ": 43100,
                "p_K": 44000,
                "p_1": 40400,  # S-reduced, decadic
                "p_2": 50300,
                "p_3": 60200,
                "p_4": 70100,
                "p_5": 80000,
            }
        )
        return coding


class AsymmetricTop(SymmetricTop):
    def __init__(
        self,
        custom_coding: Union[None, Dict[str, Union[str, int]]] = None,
        spins: Union[None, List[int], Dict[int, int]] = None,
        **params,
    ):
        super().__init__(custom_coding, spins, **params)
        # parameter checking making sure that things make sense
        A, B, C = [params.get(key, 0.0) for key in ["A", "B", "C"]]
        assert A >= B >= C

    @property
    @hyperfine_nuclei
    def param_coding(self) -> Dict[str, Union[str, int]]:
        # inherit coding from symmetric tops
        coding = super().param_coding
        coding.update(
            {
                "A": 10000,
                "B": 20000,
                "C": 30000,
            }
        )
        return coding


class SPCAT:

    __quanta_map__ = {"AsymmetricTop": 1, "SymmetricTop": -1, "LinearMolecule": -1}

    def __init__(
        self,
        T: float = 300.0,
        int_limits: List[float] = [-20.0, -5.0],
        freq_limit: float = 300.0,
        k_limit: int = 100,
        mu: List[float] = [1.0, 0.0, 0.0],
        rep: str = 'Ir',
        s_reduced: bool = True,
        q: float = 2.2415,
        weight_axis: int = 1,
        weights: List[int] = [1, 1],
        max_f_qno: int = 99,
        mol_id: int = 42,
    ):
        super().__init__()
        assert len(mu) == 3  # we need exactly three dipole moments
        # convert all values to their absolute values
        mu = list(map(abs, mu))
        # make sure the dipoles are non-zero
        assert sum(mu) != 0
        self.T = T
        self._int_limits = int_limits
        self.freq_limit = freq_limit  # this forces the setter method
        self.k_max = k_limit
        self.s_reduced = s_reduced
        self._mu = mu
        self.q = q
        self.weight_axis = weight_axis
        self._weights = weights
        self.rep = rep
        self.max_f_qno = max_f_qno
        self.mol_id = mol_id

    @classmethod
    def from_yml(cls, yml_path: str, **kwargs):
        """
        Class method to instantiate an `SPCAT` object, by
        providing a list of dipole moments, and a YAML file
        containing parameters for the simulation, i.e.
        those contained in the `.int` file for SPCAT.
        
        In essence, this is a convenience method for automatically
        generating many different catalogs of molecules, where
        the only thing that really changes is the dipole
        moment (and all the other settings are the same).

        Parameters
        ----------
        yml_path : str
            [description]
        
        Additional kwargs are used to override the loaded
        YAML settings.

        Returns
        -------
        [type]
            [description]
        """
        var_dict = routines.read_yaml(yml_path)
        var_dict.update(kwargs)
        return cls(**var_dict)

    @property
    def int_limits(self) -> List[float]:
        """
        The intensity limits of the simulation in log units.
        The values are always sorted, such that the lower limit
        is the first of the pair.

        Returns
        -------
        List[float]
            2-tuple containing the log intensity cutoffs,
            in ascending order (min, max).
        """
        return self._int_limits

    @int_limits.setter
    def int_limits(self, value: List[float]):
        value = sorted(value)
        self._int_limits = value

    @property
    def mol_id(self) -> int:
        return self._mol_id

    @mol_id.setter
    def mol_id(self, value: int):
        assert isinstance(value, int)
        self._mol_id = value

    @property
    def T(self) -> float:
        """
        Returns the temperature in kelvin used for
        the spectral simulation.

        Returns
        -------
        float
            Temperature in kelvin
        """
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        assert value > 0.0
        self._T = value

    @property
    def max_f_qno(self) -> int:
        return self._max_f_qno

    @max_f_qno.setter
    def max_f_qno(self, value: int):
        assert value > 0
        self._max_f_qno = value

    @property
    def int_limits(self) -> List[float]:
        """
        Returns the intensity limits of the simulation,
        ordered as [min, max] in units appropriate for
        the backend.

        Returns
        -------
        List[float]
            Intensity limits considered in the simulation
            as [`min_int`, `max_int`]
        """
        return self._int_limits

    @property
    def freq_limit(self) -> float:
        """
        Returns the frequency limits of the simulation,
        ordered as [min, max] in units of GHz.

        Returns
        -------
        List[float]
            Frequency limits considered in the simulation
            as [`min_freq`, `max_freq`]
        """
        return self._freq_limit

    @freq_limit.setter
    def freq_limit(self, value: float) -> None:
        assert value > 0.0
        self._freq_limit = value

    @property
    def k_max(self) -> int:
        """
        Returns the maximum value of K considered in the
        simulation.

        Returns
        -------
        int
            K max
        """
        return self._k_max

    @k_max.setter
    def k_max(self, value: int) -> None:
        assert 0 <= value
        self._k_max = value

    @property
    def mu(self) -> List[int]:
        """
        Returns the three dipole moments in debye, ordered
        along a, b, c axes.

        Returns
        -------
        List[int]
            Three-element list corresponding to the
            dipole moments along a, b, c axes in debye.
        """
        return self._mu

    @property
    def reduction(self) -> str:
        """
        Returns the Watson reduction used for the simulation.
        This is set by the `SPCAT._s_reduced` flag, which
        is `True` is using S-reduction.

        Returns
        -------
        str
            Returns "s" if `_s_reduced` is set to `True`, otherwise
            "a" for A-reduced Hamiltonian.
        """
        return "s" if self.s_reduced else "a"

    @property
    def s_reduced(self) -> bool:
        return self._s_reduced

    @s_reduced.setter
    def s_reduced(self, value: bool) -> None:
        self._s_reduced = value

    @property
    def q(self) -> float:
        return self._q

    @q.setter
    def q(self, value: float) -> float:
        assert value > 0.0
        self._q = value

    @property
    def weight_axis(self) -> int:
        __doc__ = SPCAT.weight_axis.setter.__doc__
        return self._weight_axis

    @weight_axis.setter
    def weight_axis(self, value: int) -> None:
        """
        Set the axis for statistical weighting, called IAX. According to Herb
        Pickett's documentation, the magnitude of this value corresponds to:

        1=a; 2=b; 3=c; 4= A, 2-fold top; 5=Bz, 2-fold top; 6= 3-fold top;
        7=A, E, 4-fold top; 8=B, 4-fold top; 9=5-fold top;
        10=A, E2, 6-fold top; 11=B, E1, 6-fold top).

        > For mag IAX > 3, axis is b. (See Special Considerations for Symmetric Tops)

        For the sign of this parameter:
        > If negative, use Itot basis in which the last n spins are summed to give Itot,
        > which is then combined with the other spins to give F.

        Parameters
        ----------
        value : int
            Axis used for statistical weight definition
        """
        assert abs(value) < 12
        self._weight_axis = value

    @property
    def weights(self) -> List[int]:
        return self._weights

    def format_var(self, molecule: Type[AbstractMolecule]) -> str:
        """
        Formats a string containing the .var/.par file information.
        Takes an `AbstractMolecule` as input, as it is required to
        identify the type of quantum numbers used (e.g. asymmetric top),
        as well as all the Hamiltonian encodings.

        Parameters
        ----------
        molecule : Type[AbstractMolecule]
            An instance of an `AbstractMolecule`

        Returns
        -------
        str
            Formatted string for an SPCAT .par/.var file.
        """
        data = {key: getattr(self, key) for key in ["k_max", "weight_axis"]}
        for key, value in zip(["even_weight", "odd_weight"], self.weights):
            data[key] = value
        mol_type = molecule.type
        data["quanta"] = self.__quanta_map__.get(mol_type)
        # this key is used to calculate hyperfine splitting as well
        data["quanta"] *= molecule.multiplicity
        data["reduction"] = self.reduction
        data["representation"] = -1 if self.rep=="IIIl" else 1
        data["parameters"] = str(molecule)
        return par_template.format_map(data)

    def format_int(self) -> str:
        """
        Formats a string containing the .int file information.

        Returns
        -------
        str
            SPCAT .int file contents
        """
        data = {
            key: getattr(self, key)
            for key in ["mol_id", "q", "max_f_qno", "freq_limit", "T"]
        }
        dipole_moments = ""
        for index, value in enumerate(self.mu):
            if value > 0:
                dipole_moments += f" {index + 1}  {value:.3f}\n"
        data["dipole_moments"] = dipole_moments
        for key, value in zip(["int_min", "int_max"], self.int_limits):
            data[key] = value
        return int_template.format_map(data)

    def run(self, molecule: Type[AbstractMolecule], debug: bool = False):
        """
        Abstract interface for calling an external executable
        to run the simulation.
        """
        with work_in_temp():
            var_file = self.format_var(molecule)
            int_file = self.format_int()
            for ext, contents in zip([".var", ".int"], [var_file, int_file]):
                with open(f"temp_{self.mol_id}{ext}", "w+") as write_file:
                    write_file.write(contents)
            initial_q, q_array = run_spcat(f"temp_{self.mol_id}", True, debug)
        index = np.searchsorted(q_array[0], self.T)
        # check to see if the correct value of Q was used. If not,
        # rerun SPCAT with the correct value
        if q_array[1, index] != initial_q:
            self.q = q_array[1, index]
            with work_in_temp():
                var_file = self.format_var(molecule)
                int_file = self.format_int()
                for ext, contents in zip([".var", ".int"], [var_file, int_file]):
                    with open(f"temp_{self.mol_id}{ext}", "w+") as write_file:
                        write_file.write(contents)
                initial_q, _ = run_spcat(f"temp_{self.mol_id}", False, debug)
                for file in glob("*.cat") + glob("*.out"):
                    copy2(file, "..")
        return initial_q, q_array


def infer_molecule(parameters: Dict[str, Union[str, float]]) -> Type[AbstractMolecule]:
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
    elif all([key in parameters.keys() for key in ["A-B", "B"]]) or all([key in parameters.keys() for key in ["B-C", "B"]]):
        return SymmetricTop
    else:
        return LinearMolecule


def load_molecule_yaml(
    mol_yaml: Union[str, Path]
) -> Tuple[Type[AbstractMolecule], Dict[str, str], List[float]]:
    """
    Load in the molecule specification based on a
    standardized YAML input. This function parses
    out the metadata fields and dipole moments for
    subsequent use, and returns a molecule object
    constructed with the rotational parameters.

    Parameters
    ----------
    mol_yaml : Union[str, Path]
        Path to the YAML file
        
    Returns
    ----------
    molecule : Type[AbstractMolecule]
        Appropriate rotor object
    
    metadata : Dict[str, str]
        Metadata associated with the entry

    mu : List[float]
        Three dipole moments to be passed to `SPCAT`
    """
    if isinstance(mol_yaml, str):
        mol_yaml = Path(mol_yaml)
    data = routines.read_yaml(mol_yaml)
    # standardize the key/values for SPCAT
    data = sanitize_keys(data)
    hash = routines.hash_file(mol_yaml)
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
    mu = []
    # extract the dipoles
    for key in ["u_a", "u_b", "u_c"]:
        mu.append(abs(data.get(key, 0)))
        if key in data:
            del data[key]
    unsupported = list(
        filter(lambda x: any([key in x for key in ["eps", "V", "theta"]]), data.keys())
    )
    if len(unsupported) != 0:
        raise KeyError(
            f"""Unsupported parameters in YAML; keys: {", ".join(unsupported)}"""
        )
    # infer the reduction from the keys specified
    var_kwargs = {
        "mu": mu,
        "s_reduced": False if "Delta_J" in data else True,
    }
    # look for the axis representation, if it does not appear as a parameter or
    # is not a supported representation, then choose one based on the value of 
    # Ray's asymmetry parameter
    # Case 1: Linear molecule or prolate symmetric top
    if mol_type.__name__ == "LinearMolecule" or "A-B" in data:
        var_kwargs["rep"]="Ir"
    # Case 2: Oblate symmetric top
    elif "B-C" in data:
        var_kwargs["rep"]="IIIl"
    # Case 3: Asymmetric top in Ir or IIIl representation
    elif "rep" in data and data.get("rep") in ["Ir","IIIl"]:
        var_kwargs["rep"]=data.get("rep")
        del data["rep"]
    # Case 4: Asymmetric top in Il, IIr, IIl, or IIIr representation
    elif "rep" in data:
        del data["rep"]
        var_kwargs["rep"]="IIIl" if kappa(data["A"],data["B"],data["C"])>0 else "Ir"
    # Case 5: Asymmetric top, no representation specified
    else:
        var_kwargs["rep"]="IIIl" if kappa(data["A"],data["B"],data["C"])>0 else "Ir"
    # choose the correct molecule type based on the molecular constants
    molecule = mol_type(**data)
    return (molecule, metadata, var_kwargs)


from abc import ABC, abstractmethod
from typing import Dict, List, Union, Type
from warnings import warn
from functools import wraps
from pathlib import Path
from typing import List, Dict
from subprocess import run, PIPE
from difflib import get_close_matches

from pyspectools import routines


spcat_template = """PySpecTools SPCAT input
 100  255    1    0    0.0000E+000    1.0000E+003    1.0000E+000 1.0000000000
{reduction}   {quanta}    {top}    0   {k_max}    0    {weight_axis}    {even_weight}    {odd_weight}     0   1   0
{parameters}
"""


def hyperfine_nuclei(method):
    """
    Defines a decorator that dynamically generates hyperfine
    nuclei coding, which returns the coding mapping based on
    what the user provides with respect to "chi_xx" parameters.
    """
    @wraps(method)
    def reparameterize(molecule_obj):
        coding = method(molecule_obj)
        hyperfine_names = ["chi_aa", "chi_bb", "chi_cc"]
        # only operate when there are nuclei
        if molecule_obj.num_nuclei != 0:
            for nucleus in molecule_obj.nuclei:
                for index, name in enumerate(hyperfine_names):
                    hf_code = f"{nucleus}100{index+1}0000"
                    coding[f"{name}_{nucleus}"] = hf_code
        # override with the user specified terms at the end
        coding.update(molecule_obj.custom_coding)
        return coding
    return reparameterize


class Parameter(object):
    def __init__(self, name: str, value: float, unc: float = 0.):
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


class AbstractMolecule(ABC):
    def __init__(self, custom_coding: Union[None, Dict[str, Union[str, int]]] = None, **params):
        self._nuclei = list()
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
                if number not in self._nuclei:
                    self._nuclei.append(number)
            setattr(self, key, param)
        self.param_names = list(params.keys())

    @property
    def params(self) -> Dict[str, Type[Parameter]]:
        params = {key: getattr(self, key) for key in self.param_names}
        return params

    @property
    def nuclei(self) -> List[int]:
        return self._nuclei

    @property
    def num_nuclei(self) -> int:
        return len(self._nuclei)

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

    @property
    @abstractmethod
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
                warn(f"{key} has not been implemented in {self.__class__.__name__} and was ignored.\nCloset matches are {close}")
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
    def __init__(self, custom_coding: Union[None, Dict[str, Union[str, int]]] = None, **params):
        super().__init__(custom_coding, **params)

    @property
    @hyperfine_nuclei
    def param_coding(self) -> Dict[str, Union[str, int]]:
        coding = {
            "B": 100,
            "D": 200,
            "H": 300,
            "L": 400
        }
        return coding


class SymmetricTop(LinearMolecule):
    def __init__(self, custom_coding: Union[None, Dict[str, Union[str, int]]] = None, **params):
        super().__init__(custom_coding, **params)

    @property
    @hyperfine_nuclei
    def param_coding(self) -> Dict[str, Union[str, int]]:
        # inherit coding from the linear molecule
        coding = super().param_coding
        coding.update(
            {
                "A-B": 1000,
                # main differences are in the quartic terms
                "DK": 2000,
                "DJK": 1100,
                "DJ": coding.get("D"),    # this basically sets up an alias
                "deltaJ": 40100,
                "deltaK": 41000,
                "d1": 40100,
                "d2": 50000,
                # sextic constants
                "HK": 3000,
                "HJK": 1200,
                "HKJ": 2100,
                "HJ": coding.get("H"),
            }
        )
        return coding


class AsymmetricTop(SymmetricTop):
    def __init__(self, custom_coding: Union[None, Dict[str, Union[str, int]]] = None, **params):
        super().__init__(custom_coding, **params)
        # parameter checking making sure that things make sense
        A, B, C = [params.get(key, 0.) for key in ["A", "B", "C"]]
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

    __quanta_map__ = {
        "AsymmetricTop": 1,
        "SymmetricTop": -1,
        "LinearMolecule": -1
    }

    __reduction_map__ = {
        True: "s",
        False: "a"
    }

    def __init__(self, T: float = 300., 
    int_limits: List[float] = [-10., -20.], 
    freq_limits: List[float] = [0., 500.],
    k_limit: int = 100,
    mu: List[float] = [1., 0., 0.],
    prolate: bool = True,
    s_reduced: bool = True,
    q: float = 1000.,
    weight_axis: int = 1,
    weights: List[int] = [1, 1],
    ):
        super().__init__()
        assert len(mu) == 3             # we need exactly three dipole moments
        self.T = T
        self._int_limits = int_limits
        self.freq_limits = freq_limits  # this forces the setter method
        self.k_max = k_limit
        self.s_reduced = s_reduced
        self._mu = mu
        self.q = q
        self.weight_axis = weight_axis
        self._weights = weights
        self.prolate = prolate

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
        assert value >= 0.
        self._T = value

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
    def freq_limits(self) -> List[float]:
        """
        Returns the frequency limits of the simulation,
        ordered as [min, max] in units of GHz.

        Returns
        -------
        List[float]
            Frequency limits considered in the simulation
            as [`min_freq`, `max_freq`]
        """
        return self._freq_limits

    @freq_limits.setter
    def freq_limits(self, value: List[float]) -> None:
        self._freq_limits = sorted(value)

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
        assert value > 0.
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

    def format_input(self, molecule: Type[AbstractMolecule]) -> str:
        data = {key: getattr(self, key) for key in ["k_max", "weight_axis"]}
        for key, value in zip(["even_weight", "odd_weight"], self.weights):
            data[key] = value
        mol_type = molecule.type
        data["quanta"] = self.__quanta_map__.get(mol_type)
        data["reduction"] = self.reduction
        data["top"] = 99 if self.prolate else -99
        data["parameters"] = str(molecule)
        return spcat_template.format_map(data)

    def run(self, molecule: Type[AbstractMolecule]):
        """
        Abstract interface for calling an external executable
        to run the simulation.
        """
        raise NotImplementedError

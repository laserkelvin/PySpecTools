
import pprint

import torch
import numpy as np
from uncertainties import ufloat, ufloat_fromstr
from monsterurl import get_monster
from joblib import dump, load
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

from pyspectools.models import torch_models
from pyspectools.units import kappa, inertial_defect

"""
This module defines a set of classes for high-level interaction with
the user. These classes provide a set of methods that will assist
the user in performing inference with PyTorch models by taking care
of some aspects (e.g. plotting, density estimation) in the background.
"""

class SpecConstants:
    """
    Class representing experimental parameters to be fed into the
    `MoleculeDetective` model. The user provides a set of constants
    as input, and the main purpose of this class is to help manage
    experimental uncertainties.
    """
    def __init__(self, A: str, B: str, C: str, u_a=ufloat(0., 3.), u_b=ufloat(0., 3.), u_c=ufloat(0., 3.), **kwargs):
        """
        Create a `SpecConstants` object based on experimental parameters. The minimal
        input contains a set of rotational constants, A, B, and C, and optional args
        are the dipole moments along each axis. In the case of the rotational constants,
        the expected inputs are actually strings (in MHz), representing value(uncertainty),
        for example:
        
        A = "5623.562(102)"
        
        This reflects the same notation as what SPFIT provides. The advantage of being
        able to provide uncertainties is the ability to set priors for poorly or not
        determined parameters. For example, if A is not determined, we provide an uniformative
        prior by setting a very large uncertainty for A:
        
        A = "8032(6000)"
        
        The `generate_samples` method will take uncertainties into account,
        and prepare samples for variational inference by the `MoleculeDetective` model.
        
        By default, the dipole moments are based on uninformative priors. If you know
        the value of one axis, or at least have a good idea of what it may be, or if you want
        to deactivate certain axes, you can specify them; the example below sets the
        b and c axes to zero, while "turning on" the a-dipole moment:
        
        from uncertainties import ufloat
        
        u_a = ufloat(1., 0.5)
        u_b = ufloat(0., 0.)
        u_c = ufloat(0., 0.)

        Parameters
        ----------
        A, B, C : str
            Principal rotational constants. The expected input for these are
            string representations of value(uncertainty), for example:
            "5962.3213(102)"

        u_a, u_b, u_c : ufloat, optional
            Dipole moments as `ufloat` objects, by default ufloat(0., 3.), and
            correspond to uninformative priors.
        """
        self.A = ufloat_fromstr(A)
        self.B = ufloat_fromstr(B)
        self.C = ufloat_fromstr(C)
        self.u_a = u_a
        self.u_b = u_b
        self.u_c = u_c
        # generate a name
        self.__name__ = get_monster()
        # Propagate uncertainty of kappa and delta based
        # on experimental constants
        self.kappa = kappa(self.A, self.B, self.C)
        self.delta = inertial_defect(self.A, self.B, self.C)
        # override defaults with user-specified values
        if kwargs:
            self.__dict__.update(**kwargs)
    
    def __repr__(self):
        return pprint.PrettyPrinter(indent=4).pformat(self.__dict__)
    
    def __call__(self, N=1000):
        return self.generate_samples(N)
    
    def save(self, path=None):
        """
        Save the molecule to disk using Pickle in `joblib`. 

        Parameters
        ----------
        path : str, optional
            Name to save the molecule to, not including the
            extension ".pkl", by default None
        """
        if path is None:
            path = f"{self.__name__}.pkl"
        dump(self, path)
    
    @classmethod
    def load(cls, path):
        return load(path)
        
    def generate_samples(self, N: int):
        """
        Function to generate samples of spectroscopic parameters, based on
        "diagonal" Gaussians. The nominal value and standard deviations of
        each parameter are used parameterize a Gaussian, and `N` random 
        samples are drawn. In the case of the dipole moments, we take the
        absolute value of the samples, and delta and kappa are recalculated
        based on the drawn A, B, C.
        
        TODO - Make this code look cleaner; there must be a smarter way to sample

        Parameters
        ----------
        N : int
            Number of samples to generate.

        Returns
        -------
        samples
            2D np.ndarray, where columns correspond to parameter, and rows
            are samples
        """
        A = np.abs(np.random.normal(self.A.nominal_value, self.A.std_dev, size=(N)))
        B = np.abs(np.random.normal(self.B.nominal_value, self.B.std_dev, size=(N)))
        C = np.abs(np.random.normal(self.C.nominal_value, self.C.std_dev, size=(N)))
        u_a = np.abs(np.random.normal(self.u_a.nominal_value, self.u_a.std_dev, size=(N)))
        u_b = np.abs(np.random.normal(self.u_b.nominal_value, self.u_b.std_dev, size=(N)))
        u_c = np.abs(np.random.normal(self.u_c.nominal_value, self.u_c.std_dev, size=(N)))
        # calculate derived parameters
        asymmetry = kappa(A, B, C)
        delta = inertial_defect(A, B, C)
        samples = np.vstack([A, B, C, asymmetry, delta, u_a, u_b, u_c]).T
        # We need to make sure physics is constrained: A >= B >= C
        (valid_mask,) = np.where((A >= B) & (B >= C))
        return samples[valid_mask]
    

class MoleculeResult:
    
    func_encoding = [
        "Aliphatic",
        "Allene",
        "Vinyl",
        "Alkyne",
        "Carbonyl (General)",
        "Carbonyl (α-nitrogen)",
        "Carbonyl (α-carbon)",
        "Aldehyde",
        "Amide",
        "Ketone",
        "Ether",
        "Amine",
        "Amino acid",
        "Nitrate",
        "Nitrile",
        "Isonitrile",
        "Nitro",
        "Alcohol",
        "Alcohol (Carboxylic acid)",
        "Enol",
        "Phenol",
        "Peroxide",
        "Aromatic sp2 carbon"
    ]
    
    def __init__(self, eigenspectrum, formulas, functional_groups):
        self.eigenspectrum = eigenspectrum
        self.formulas = pd.DataFrame(formulas, columns=["H", "C", "O", "N"])
        self.functional_groups = pd.DataFrame(functional_groups, columns=self.func_encoding)
    
    def analyze(self, q=(0.025, 0.5, 0.975)):
        """
        Convenience function to compute some summary statistics, and make interactive
        plotly figures.

        Parameters
        ----------
        q : tuple, optional
            [description], by default (0.025, 0.5, 0.975)

        Returns
        -------
        fig
            Plotly Figure object
        
        results
            dict containing summary statistics of the formula/functional
            predictions.
        """
        # covariance matrix of the formulas and functional group predictions
        formula_covar = np.cov(self.formulas, rowvar=False)
        functional_covar = np.cov(self.functional_groups, rowvar=False)
        # calculate the quantile ranges to report as marginal uncertainties
        # by default, the specified range is the median, and the edges are
        # the 95% highest posterior density
        formula_q = np.quantile(self.formulas, q, axis=0)
        functional_q = np.quantile(self.functional_groups, q, axis=0)
        # package the statistical summary
        results = {
            "formula": {"covariance": formula_covar, "quantile": formula_q},
            "functional": {"covariance": functional_covar, "quantile": functional_q}
        }
        
        fig = go.Figure()
        for index, atom in enumerate(["H", "C", "O", "N"]):
            fig.add_trace(
                go.Violin(
                    x=[atom,] * len(self.formulas),
                    y=self.formulas[atom],
                    name=atom,
                    meanline_visible=True,
                    opacity=0.6,
                )
            )
        fig.update_layout(title_text="Predicted formula", xaxis_title="Atom", yaxis_title="Number")
        return fig, results


class MoleculeDetective:
    def __init__(self, weights_path=None, device="cpu", **kwargs):
        self.model = torch_models.VarMolDetect.load_weights(weights_path, device, **kwargs)
        # set to inference mode
        self.model.eval()
        # For taking care of moving data around
        self.device = self.model.device
    
    def __call__(self, specconst_obj: "SpecConstants", composition: int, N=1000):
        return self.run_inference(specconst_obj, composition, N)
    
    def run_inference(self, specconst_obj: "SpecConstants", composition=None, N=1000):
        """
        Use a pre-trained PyTorch model to perform inference, conditional on the
        experimental constants and the expected composition. This framework can
        be used to account for various forms of uncertainties, and the default
        behavior is to provide the minimum amount of information. For example,
        the `composition` argument can be provided as an `int` type representing:
        
        [0: hydrocarbon, 1: oxygen-bearing, 2: nitrogen-bearing, 3: ON-bearing]
        
        By default, `composition` is None; the case where we don't know what
        the composition is, in which case we will randomly try all four compositions.
        
        This implementation is set up to take advantage of performance in `torch`. Rather
        than repeatedly call a function a la MCMC sampling, we simply pass an entire tensor
        of all the samples so that `torch` can run without Python interaction. You can
        think of each row as a "minibatch".
        
        The constants are converted to GHz, and therefore expected as MHz.

        Parameters
        ----------
        specconst_obj : [type]
            `SpecConstants` object, which will generate random samples
            based on the experimental uncertainties.
        composition : int or None (default), optional
            The expected composition of the molecule. When `composition` is
            an integer, inference is performed conditional on the specific
            composition; the definitions are provided in the docstring.
            If `None` (default), we have no prior knowledge of the composition,
            and will randomly test all four.
            
        N : int, optional
            Number of samples to run, by default 1000

        Returns
        -------
        eigenspectrum, formula, functional
            NumPy arrays corresponding to the predicted eigenspectrum,
            molecular formula, and functional groups present. `functional`
            is output as log sigmoid by the model, and this function
            returns the exponential to get back sigmoid likelihoods.
        """
        samples = specconst_obj(N)
        # convert to GHz
        samples[:,:3] /= 1000.
        if composition is None:
            composition = np.random.randint(low=0, high=4, size=len(samples))
            compositions = np.zeros((len(samples), 4))
            compositions[:,composition] = 1
        else:
            # generate one-hot encoding for the composition
            compositions = np.zeros((len(samples), max(4, composition)))
            compositions[:, composition] = 1
        # Type cast Tensors into float32 from NumPy arrays, and move to
        # CPU or GPU
        samples = torch.from_numpy(samples).float().to(self.device)
        compositions = torch.from_numpy(compositions).float().to(self.device)
        # Run the model
        with torch.no_grad():
            pred, latent = self.model(samples, compositions)
        # unpack predictions
        eigenspectrum, formula, functional = pred
        result = MoleculeResult(eigenspectrum.numpy(), formula.numpy(), functional.numpy())
        return result

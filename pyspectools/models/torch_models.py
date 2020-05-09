
from pathlib import Path
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class GenericModel(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_weights(cls, weights_path=None, device=None, **kwargs):
        """
        Convenience method for loading in the weights of a model.
        Basically initializes the model, and wraps a `torch.load`
        with automatic cuda/cpu detection.
        
        Parameters
        ----------
        weights_path : str
            String path to the trained weights of a model; typically
            with extension .pt
        
        device : str
            String reference to the target device, either "cpu", "cuda",
            or a specific CUDA device (e.g. "cuda:0"). If None (default)
            the model will be loaded onto a GPU if available, otherwise
            a CPU.
            
        kwargs are passed into the creation of the model, allowing you
        to set different parameters.
        
        Returns
        -------
        model
            Instance of the PyTorch model with loaded weights
        """
        # default location for weights is the package directory,
        # along with the model name
        if weights_path is None:
            pkg_dir = Path(__file__).parent
            weights_path = pkg_dir.joinpath(f"{cls.__name__}.pt")
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        model = cls(**kwargs)
        model.device = device
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model

    def init_layers(self, weight_func=None, bias_func=None):
        """
        Function that will initialize all the weights and biases of
        the model layers. This function uses the `apply` method of
        `Module`, and so will only work on layers that are contained
        as children.
        
        Parameters
        ----------
        weight_func : `nn.init` function, optional
            Function to use to initialize weights, by default None
            which will default to `nn.init.xavier_normal`
        bias_func : `nn.init` function, optional
            Function to use to initialize biases, by default None
            which will default to `nn.init.xavier_uniform`
        """
        # Apply initializers to all of the Module's children with `apply`
        self.apply(self._initialize_wb)

    def _initialize_wb(self, layer: nn.Module):
        """
        Static method for applying an initializer to weights
        and biases. If a layer is passed without weight and
        bias attributes, this function will effectively ignore it.
        
        Parameters
        ----------
        layer : `nn.Module`
            Layer that is a subclass of `nn.Module`
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.uniform_(layer.bias.data, a=-1., b=1.)
        elif isinstance(layer, nn.LSTM):
            torch.nn.init.xavier_uniform_(layer.weight_hh_l0)
            torch.nn.init.xavier_uniform_(layer.weight_ih_l0)
            if layer.bias_ih_l0 is not None:
                torch.nn.init.zeros_(layer.bias_ih_l0)
            if layer.bias_hh_l0 is not None:
                torch.nn.init.zeros_(layer.weight_hh_l0)
        if isinstance(layer, nn.BatchNorm1d):
            torch.nn.init.ones_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias.data)

    def __len__(self):
        return self.get_num_parameters()

    def get_num_parameters(self) -> int:
        """
        Calculate the number of parameters contained within the model.
        
        Returns
        -------
        int
            Number of trainable parameters
        """
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @abstractmethod
    def compute_loss(self):
        pass

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Private method for scale/shift operation on a unit Gaussian
        (N~[0,1]) using the parameterized mu and logvar in a variational
        model. Returns the latent encoding based on these values.

        Parameters
        ----------
        mu : torch.Tensor
            Tensor of Gaussian centers.
        logvar : torch.Tensor
            Tensor of log variance

        Returns
        -------
        z
            Torch Tensor corresponding to the latent embedding
        """
        std = logvar.exp().sqrt()
        eps = torch.autograd.Variable(torch.randn_like(std))
        return eps.mul(std).add(mu)


class VariationalSpecDecoder(GenericModel):

    """
    Uses variational inference to capture the uncertainty
    with respect to Coulomb matrix eigenvalues. Instead of
    using dropout, this model represents uncertainty via a
    probabilistic latent layer.
    """

    __name__ = "VariationalSpecDecoder"

    def __init__(
        self,
        latent_dim=14,
        output_dim=30,
        alpha=0.8,
        dropout=0.2,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        tracker=True,
    ):
        super().__init__()
        self.mu_dense = nn.Linear(12, latent_dim)
        self.logvar_dense = nn.Linear(12, latent_dim)
        # output should all be positive
        self.spec_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(128, 256),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
        )
        self.input_drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor):
        """
        Inputs for this model is a single Tensor, where each row
        is 12 elements long (8 constants, one-hot encoding for
        composition). The idea behind this is to predict the
        eigenspectrum conditional on the molecular composition.

        Parameters
        ----------
        X : torch.Tensor
            Tensor containing spectroscopic constants, and
            one-hot encoding of the composition.

        Returns
        -------
        output, mu, logvar
            The predicted eigenspectrum, and the latent parameters
            mu and logvar
        """
        mu, logvar = self.mu_dense(X), -F.relu(self.logvar_dense(X))
        z = self._reparametrize(mu, logvar)
        output = self.spec_decoder(z)
        return output, mu, logvar

    def compute_loss(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Calculate the loss of this model as the combined prediction error
        and KL-divergence from the approximate posterior.

        Parameters
        ----------
        X : torch.Tensor
            Combined tensor of the spectroscopic constants and the one-hot
            encoded composition.
            
        Y : torch.Tensor
            Target eigenspectrum

        Returns
        -------
        torch.Tensor
            Joint loss of MSE and KL divergence
        """
        pred_Y, mu, logvar = self.forward(X)
        accuracy = F.mse_loss(pred_Y, Y, reduction="sum")
        var = logvar.exp()
        # The summation is performed over the encoding length, as according to Kingma
        KL = -0.5 * torch.sum(1 + 2.0 * logvar - mu.pow(2.0) - var.pow(2.0))
        return accuracy + KL


class VariationalDecoder(GenericModel):
    
    """
    This model uses the intermediate eigenspectrum to calculate a
    latent embedding that is then used to predict the molecular
    formula and functional groups. You can think of the first action
    as "re-encoding", but the driving principle is that an eigenspectrum
    could map onto various structures, even when conditional on the
    composition.
    """
    
    __name__ = "VariationalDecoder"
    
    def __init__(
        self,
        latent_dim=14,
        eigen_length=30,
        nclasses=23,
        alpha=0.8,
        dropout=0.2,
        loss_func=None,
        param_transform=None,
        tracker=True,
    ):
        super().__init__()
        self.mu_dense = nn.Linear(eigen_length + 4, latent_dim)
        self.logvar_dense = nn.Linear(eigen_length + 4, latent_dim)
        self.formula_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(64, 4),
            nn.ReLU(inplace=True),
        )
        self.functional_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(128, 256),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(256, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(alpha),
            nn.Linear(64, nclasses),
            nn.LogSigmoid(),
        )

    def forward(self, X: torch.Tensor):
        """
        Perform a forward pass of the VariationalDecoder model.
        This takes the concatenated input of the eigenspectrum and
        the one-hot composition, produces a latent embedding that
        is then used to predict the formula and functional group
        classification.

        Parameters
        ----------
        X : torch.Tensor
            [description]

        Returns
        -------
        formula_output
            Nx4 tensor corresponding to the number of atoms
            in the [H,C,O,N] positions.
        
        functional_output
            Nx23 tensor corresponding to multilabel classification,
            provided as log sigmoid.
        
        mu, logvar
            Latent variables of the variational layer
        """
        mu, logvar = self.mu_dense(X), -F.relu(self.logvar_dense(X))
        # generate latent representation
        z = self._reparametrize(mu, logvar)
        formula_output = self.formula_decoder(z)
        functional_output = self.functional_classifier(z)
        return formula_output, functional_output, mu, logvar

    def compute_loss(
        self, X: torch.Tensor, formula: torch.Tensor, groups: torch.Tensor
    ):
        """
        Calculate the joint loss of this model. This corresponds to the sum
        of three components: a KL-divergence loss for the variational layer,
        a formula prediction accuracy as the MSE loss, and the BCE loss for the
        multilabel classification for the functional group prediction.

        Parameters
        ----------
        X : torch.Tensor
            [description]
        formula : torch.Tensor
            Length of the formula encoding, typically 4 [H,C,O,N]
        groups : torch.Tensor
            Length of the functional groups encoding.
        """
        pred_formula, pred_func, mu, logvar = self.forward(X)
        # Predict atom number
        accuracy = F.mse_loss(pred_formula, formula, reduction="sum")
        # Multilabel classification
        classification = F.binary_cross_entropy_with_logits(pred_func, groups, reduction="sum")
        # calculate the divergence term
        var = logvar.exp()
        # The summation is performed over the encoding length, as according to Kingma
        KL = -0.5 * torch.sum(1 + 2.0 * logvar - mu.pow(2.0) - var.pow(2.0))
        return accuracy + KL + classification


class VarMolDetect(GenericModel):

    """
    Umbrella model that encapsulates the full set of variational
    models. The premise is to more or less try to do end-to-end
    learning, and should meet the user half-way in terms of
    usability. The `forward` method takes the spectroscopic constants
    and the molecular composition as separate inputs, and performs
    the concatenation prior to any calculation. The composition
    is reused by the `VariationalDecoder` model.
    """

    __name__ = "VariationalMoleculeDetective"

    def __init__(
        self,
        eigen_length=30,
        latent_dim=14,
        nclasses=23,
        alpha=0.8,
        dropout=0.2,
        tracker=True,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(8)
        self.spectrum_decoder = VariationalSpecDecoder(latent_dim, eigen_length, alpha, dropout=dropout)
        self.joint_decoder = VariationalDecoder(
            latent_dim, eigen_length, nclasses, alpha, dropout=dropout
        )
        self.input_dropout = nn.Dropout(dropout)
        for name, parameter in self.named_parameters():
            if "logvar" in name and "weight" in name:
                # nn.init.uniform_(parameter, a=-10., b=-8.)
                nn.init.kaiming_normal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
            elif "weight" in name and "norm" not in name:
                nn.init.kaiming_normal_(parameter)
            elif "weight" in name and "norm" in name:
                nn.init.ones_(parameter)

    def forward(self, constants: torch.Tensor, composition: torch.Tensor):
        # This mask ensures that predictions of formulae are appropriate
        # for the specified composition. The conditional estimation alone
        # was okay, but could sometimes still predict formulae it shouldn't
        comp_mask = torch.FloatTensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 1]
            ]
        ).to(constants.device)
        comp_encoding = composition.argmax(dim=-1)
        masks = torch.autograd.Variable(comp_mask[comp_encoding], requires_grad=True)
        # Run batch norm on A,B,C
        constants = self.input_dropout(self.norm(constants))
        # concatenate the inputs
        X = torch.cat((constants, composition), dim=-1)
        # compute the eigenspectrum conditional on composition
        eigen, eigen_mu, eigen_logvar = self.spectrum_decoder(X)
        # run the decoders to predict properties, conditional on the composition
        eigen_composition = torch.cat((eigen, composition), dim=-1)
        formula, functionals, decode_mu, decode_logvar = self.joint_decoder(
            eigen_composition
        )
        # Remove predictions of atoms that don't belong
        formula = formula * masks
        return (
            (eigen, formula, functionals),
            (eigen_mu, eigen_logvar, decode_mu, decode_logvar),
        )

    def compute_loss(self, constants, composition, eigenspectrum, formula, functionals):
        # run through the models
        predictions, latents = self.forward(constants, composition)
        # unpack the predictions, and compute their losses
        pred_eigen, pred_formula, pred_func = predictions
        # for regression, we take the log10 for stability; everything done in place
        prediction_loss = F.mse_loss(pred_eigen, eigenspectrum, reduction="mean")
        prediction_loss.add_(
            F.mse_loss(pred_formula, formula, reduction="mean")
        )
        prediction_loss.add_(
            F.binary_cross_entropy_with_logits(pred_func, functionals, reduction="mean")
        )
        # now for the variational losses
        eigen_mu, eigen_logvar, decode_mu, decode_logvar = latents
        eigen_var = eigen_logvar.exp()
        decode_var = decode_logvar.exp()
        # The summation is performed over the encoding length, as according to Kingma
        kl_loss = -0.5 * torch.sum(
            1 + 2.0 * eigen_logvar - eigen_mu.pow(2.0) - eigen_var.pow(2.0)
        )
        kl_loss /= constants.size(0) * eigen_logvar.size(0)
        kl_loss.add_(
            -0.5
            * torch.sum(
                1 + 2.0 * decode_logvar - decode_mu.pow(2.0) - decode_var.pow(2.0)
            )
        )
        kl_loss /= constants.size(0) * decode_logvar.size(0)
        return prediction_loss + kl_loss
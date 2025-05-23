"""
This file contains the implementation of deep learning models for regression tasks.
"""
from typing import Optional

import torch.nn
import numpy as np
from epiuc.uncertainty.base import BaseDNN
from epiuc.utils.general import (
    construct_mlp,
    validate_mlp_params,
    validate_predict_input,
    set_model_to_eval,
    validate_quantiles,
    set_seeds,
)
from epiuc.utils.loss_functions import quantile_loss, sigma_loss


class MLP(BaseDNN):
    def __init__(
        self,
        input_shape: int,
        n_layers: int = 1,
        num_neurons: int = 50,
        dropout_rate: float = 0,
        batch_norm: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initializes the MLP object.
        :param input_shape: The shape of the input data.
        :param n_layers: The number of layers in the MLP. The final layer is not included in this count.
        :param num_neurons: The number of neurons in each layer.
        :param dropout_rate: The dropout rate to be used in the MLP.
        :param batch_norm: Whether to use batch normalization or not.
        :param random_state: The random state to be used for the MLP. This is used to ensure that the MLP is reproducible.
        """

        if random_state:
            set_seeds(random_state)

        validate_mlp_params(
            n_layers, num_neurons, dropout_rate, input_shape, 1
        )  # 1 output for regression
        model = construct_mlp(
            input_features=input_shape,
            n_outputs=1,
            n_layers=n_layers,
            n_neurons=num_neurons,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
        )

        super().__init__(model=model, random_state=random_state)

        self.name = "MLP_Regression"

    def loss_function(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for the MLP uncertainty. This is the mean squared error loss function.
        :param output: The models output.
        :param target: The target value.
        :param kwargs: Additional arguments. not used.
        :return: The loss value.
        """
        return torch.nn.functional.mse_loss(output, target)

    @torch.no_grad()
    def _predict(self, X: torch.Tensor | np.ndarray, raw_output=False) -> torch.Tensor:
        """
        This method is used to get the mean prediction of the uncertainty.
        :param X: input data
        :param raw_output: not used.

        """
        set_model_to_eval(self.model)

        X = validate_predict_input(X, self.device)

        return self(X)


class MLP_Sigma_Softplus(torch.nn.Module):
    """
    This class is used to transform the mean and sigma of an MLP modelling a  Gaussian distribution.
    """

    def __init__(self):
        super(MLP_Sigma_Softplus, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies softplus to the log sigma and concatenates it with the mean.
        :param x: input data
        :return: mean and sigma of the Gaussian distribution
        """
        mu, sigma = torch.split(x, 1, dim=-1)

        # +1e-6 to make sure sigma is never zero
        # sigma = torch.nn.functional.softplus(logsigma) + 1e-6
        sigma = torch.exp(sigma)

        output = torch.cat([mu, sigma], dim=-1)
        return output


class MLP_Sigma(MLP):
    """
    This class is used to implement an MLP uncertainty that outputs the mean and sigma of a Gaussian distribution.
    """

    def __init__(
        self,
        input_shape: int,
        n_layers: int = 1,
        num_neurons: int = 50,
        dropout_prob: float = 0,
        batch_norm: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initializes the MLP_Sigma object.
        :param input_shape: The shape of the input data.
        :param n_layers: The number of layers in the MLP.
        :param num_neurons: The number of neurons in each layer.
        :param dropout_prob: The dropout rate to be used in the MLP.
        :param batch_norm: Whether to use batch normalization or not.
        :param random_state: The random state to be used for the MLP. This is used to ensure that the MLP is reproducible.
        """
        super().__init__(
            input_shape=input_shape,
            n_layers=n_layers,
            num_neurons=num_neurons,
            dropout_rate=dropout_prob,
            batch_norm=batch_norm,
            random_state=random_state,
        )

        # Replace the last layer with a linear layer with 2 outputs (mean and sigma)
        self.model = torch.nn.Sequential(
            *list(self.model.children())[:-1],
            torch.nn.Linear(in_features=num_neurons, out_features=2, bias=True),
            MLP_Sigma_Softplus(),
        )
        self.name = "MLP_Sigma"
        self.sigma = True
        self.uncertainty_aware = True

    def loss_function(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for the MLP uncertainty. This is the negative log likelihood of the Gaussian distribution.
        :param output: The models output.
        :param target: The target value.
        :param kwargs: Additional arguments. not used.
        :return: The nll of Gaussian distribution.
        """
        # Split the output into mean and sigma
        mu, sigma = torch.split(output, 1, dim=-1)

        return sigma_loss(mu, sigma, target)

    @torch.no_grad()
    def _predict(self, X: torch.Tensor | np.ndarray, raw_output=False) -> torch.Tensor:
        """
        This method is used to get the mean and sigma of the uncertainty.
        :param X: input data
        :param raw_output: not used.
        :return: mean and sigma of the Gaussian distribution
        """
        set_model_to_eval(self.model)

        X = validate_predict_input(X, self.device)

        mu, sigma = self(X).split(1, dim=-1)
        return mu, sigma


class MLP_Quantile(MLP):
    """
    This class is used to implement an MLP uncertainty that learns quantiles.
    """

    def __init__(
        self,
        input_shape: int,
        quantiles: list[float],
        n_layers: int = 1,
        num_neurons: int = 50,
        dropout_rate: float = 0,
        batch_norm: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initializes the MLP_Quantile object.
        :param input_shape: The shape of the input data.
        :param n_layers: The number of layers in the MLP.
        :param num_neurons: The number of neurons in each layer.
        :param dropout_rate: The dropout rate to be used in the MLP.
        :param batch_norm: Whether to use batch normalization or not.
        :param random_state: The random state to be used for the MLP. This is used to ensure that the MLP is reproducible.
        """
        super().__init__(
            input_shape=input_shape,
            n_layers=n_layers,
            num_neurons=num_neurons,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            random_state=random_state,
        )

        # Validate the quantiles
        validate_quantiles(quantiles)

        # Remove the last layer and replace it with a linear layer with len(quantiles) outputs
        self.model = torch.nn.Sequential(
            *list(self.model.children())[:-1],
            torch.nn.Linear(
                in_features=num_neurons, out_features=len(quantiles), bias=True
            ),
        )

        self.name = "MLP_Quantiles"

        self.quantiles = quantiles

        self.uncertainty_aware = False

    def loss_function(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for the MLP uncertainty. This is the quantile loss function.
        :param output: The models output.
        :param target: The target value.
        :param kwargs: Additional arguments. not used.
        :return: The quantile loss value.
        """
        return quantile_loss(output, target, self.quantiles)

    @torch.no_grad()
    def _predict(self, X: torch.Tensor | np.ndarray, raw_outputs=False) -> torch.Tensor:
        """
        This method is used to get the quantiles of the uncertainty.
        :param X: input data.
        :param raw_outputs: not used.
        :return: The quantiles of the uncertainty.
        """
        set_model_to_eval(self.model)

        X = validate_predict_input(X, self.device)

        return self(X)

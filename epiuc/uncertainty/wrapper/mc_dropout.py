from typing import Union

import torch
import numpy as np
from copy import deepcopy

from epiuc.uncertainty.base import BaseDNN
from epiuc.utils.general import validate_predict_input
from epiuc.utils.loss_functions import sigma_loss
from epiuc.utils.uncertainty_decompositions import (
    uncertainty_decomposition_entropy,
    uncertainty_decomposition_mus_sigmas,
)

"""
This file contains the implementation of the MC Dropout uncertainty for classification and regression tasks.
"""


def _insert_dropout(model: torch.nn.Module, p: float):
    """
    Insert dropout layers to the uncertainty with given probability p.
    :param model: uncertainty to insert dropout layers
    :param p: dropout probability
    :return: uncertainty with dropout layers
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.GELU):
            model._modules[name] = torch.nn.Sequential(module, torch.nn.Dropout(p))
        else:
            _insert_dropout(module, p)
    return model


def validate_dropout_model(model, p):
    """
    Insert dropout layers to the uncertainty with given probability p, if it does not contain any dropout probability.
    :param model: uncertainty to insert dropout layers
    :param p: dropout probability
    :return: uncertainty with dropout layers
    """
    model = deepcopy(model)
    if not any([isinstance(module, torch.nn.Dropout) for module in model.modules()]):
        print(
            f"No dropout layer found in the given uncertainty. "
            f"Inserting dropout layers with probability {p}"
        )
        model = _insert_dropout(model, p)
    return model


class MC_Classifier(BaseDNN):
    """
    Monte Carlo Dropout uncertainty for classification tasks.
    This uncertainty uses Monte Carlo Dropout to estimate the uncertainty of the predictions.
    The uncertainty is trained with dropout layers and the predictions are made by conducting multiple forward passes from the dropout layers.
    """

    def __init__(
        self,
        base_model: BaseDNN,
        n_iterations: int,
        dropout_prob: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the MC Dropout uncertainty for classification with given base_model.
        :param base_model: classification uncertainty
        :param n_iterations: MC inference iterations
        :param dropout_prob: dropout probability
        :param random_state: random state for reproducibility
        """

        model = validate_dropout_model(base_model, dropout_prob)

        super(MC_Classifier, self).__init__(
            model=model,
            random_state=random_state,
        )
        # Class params
        self.name = f"MC_Classification_on_{base_model.name}"
        self.n_iterations = n_iterations
        self.uncertainty_aware = True

    def loss_function(self, output, target, **kwargs):
        """
        Calculate the cross entropy loss for the given output and target.
        :param output: The uncertainty output
        :param target: The true value
        :param kwargs: additional arguments not used here
        :return: cross entropy loss
        :return:
        """
        return torch.nn.functional.cross_entropy(output, target)

    def _predict(
        self, X: torch.Tensor | np.ndarray, raw_output=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the probabilities and epistemic uncertainty of the given input X based on MC iterations.
        :param X: input data
        :param raw_output: if True, return total_uncertainty, aleatoric uncertainty,,epistemic_uncertainty along with the individual probabilities of the forward passes, otherwise return the mean probabilities and epistemic uncertainty.
        :return: mean probabilities and epistemic or probabilities, total_uncertainty, aleatoric uncertainty, epistemic_uncertainty
        """
        # Convert to tensor
        validate_predict_input(X, self.device)

        # Activate dropout
        self.train(True)
        probs = torch.stack(
            tensors=[
                torch.nn.functional.softmax(self(X), dim=-1)
                for _ in range(self.n_iterations)
            ],
            dim=1,
        )

        probabilities = probs.mean(dim=1)

        (
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = uncertainty_decomposition_entropy(probs)

        if raw_output:
            return probs, torch.stack(
                [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty], dim=1
            )
        return probabilities, epistemic_uncertainty.reshape((-1, 1))


class MC_Regression(BaseDNN):
    """
    MC Dropout uncertainty for regression tasks.
    This uncertainty uses Monte Carlo Dropout to estimate the uncertainty of the predictions.
    The uncertainty is trained with dropout layers and the predictions are made by conducting multiple forward passes from the dropout layers.

    Notice: This uncertainty can only give the epistemic uncertainty, if the base_model predicts mean and sigma of Gaussian uncertainty.
    Additionally we assume that the sigma is positive.

    """

    def __init__(
        self,
        base_model: BaseDNN,
        n_iterations: int,
        dropout_prob: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the MC Dropout uncertainty for regression with given base_model.
        :param base_model: regression uncertainty
        :param n_iterations: MC inference iterations
        :param dropout_prob: dropout probability
        :param random_state: random state for reproducibility
        """

        model = validate_dropout_model(base_model.model, dropout_prob)

        super(MC_Regression, self).__init__(
            model=model,
            random_state=random_state,
        )

        # Class params
        self.name = f"MC_Regression_on_{base_model.name}"
        self.n_iterations: int = n_iterations
        self.sigma = base_model.sigma
        self.uncertainty_aware = True

    def loss_function(self, output, target, **kwargs):
        """
        Depending on base_model either calculate the MSE or the negative log likelihood of Gaussian Model.
        If the output has only one dimension, the MSE is calculated.
        If the output has two dimensions, the negative log likelihood of Gaussian Model is calculated.
        :param output: The uncertainty output.
        :param target: The target values.
        :param kwargs: additional arguments not used here
        :return: MSE or negative log likelihood of Gaussian Model
        """
        if self.sigma:
            mu, sigma = torch.split(output, 1, dim=-1)

            return sigma_loss(mu, sigma, target)
        return torch.nn.functional.mse_loss(output, target)

    def _predict(
        self, X: torch.Tensor | np.ndarray, raw_output: bool = False
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Predict the mean and sigma of the given input X based on MC iterations.
        :param X: input data
        :param raw_output: if True, return total_uncertainty, aleatoric uncertainty,,epistemic_uncertainty along with the individual mus/sigmas of the forward passes, otherwise return the mean mus and epistemic uncertainty.
        :return: mean mus and epistemic uncertainty or mus, sigmas, total_uncertainty, aleatoric uncertainty,epistemic_uncertainty if raw_output is True
        """
        X = validate_predict_input(X, self.device)
        self.train(True)
        with torch.no_grad():
            output = torch.stack([self(X) for _ in range(self.n_iterations)], dim=1)

            if self.sigma:
                mus, sigma = torch.split(output, 1, dim=-1)

                y_pred = mus.mean(dim=1)

                (
                    total_uncertainty,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                ) = uncertainty_decomposition_mus_sigmas(mus, sigma)

                if raw_output:
                    return (
                        output,
                        torch.stack(
                            [
                                total_uncertainty,
                                aleatoric_uncertainty,
                                epistemic_uncertainty,
                            ],
                            dim=1,
                        ),
                    )
                else:
                    return y_pred, epistemic_uncertainty

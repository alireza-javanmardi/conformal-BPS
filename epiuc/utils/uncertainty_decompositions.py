"""
Uncertainty Decomposition Functions
"""
import numpy as np
import torch
from torch import digamma

#### Uncertainty Decomposition Functions used by MC and Ensemble ####
def uncertainty_decomposition_entropy(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to compute the uncertainty decomposition based on entropy of the individual probabilities.
    :param probabilities: the individual probabilities of an ensemble of multiple forward passes.
        The shape should be (n_samples, n_models, n_classes)
    :return: total uncertainty, aleatoric uncertainty, epistemic uncertainty
    """
    min_real_value = 1e-30
    # replace 0 with a small value to avoid log(0)
    probabilities = torch.where(
        probabilities == 0,
        torch.full_like(probabilities, min_real_value),
        probabilities,
    )
    # compute entropy decomposition of the probabilities
    total_uncertainty = -1 * (
        probabilities.mean(dim=1) * torch.log2(probabilities.mean(dim=1))
    ).sum(dim=-1).div(np.log2(probabilities.shape[-1]))

    aleatoric_uncertainty = (
        (-1 * (probabilities * torch.log2(probabilities))).sum(dim=-1).mean(dim=-1)
    ).div(np.log2(probabilities.shape[-1]))

    # Due to numerical instability, the aleatoric uncertainty can be greater than the total uncertainty.
    # while this then yields valued around -(1e-10), we clip the values to 0.
    epistemic_uncertainty = (total_uncertainty - aleatoric_uncertainty).clip(0, 1)

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def uncertainty_decomposition_mus_sigmas(mus, sigmas):
    """
    Function to compute the uncertainty decomposition based on the mean and standard deviation of the individual predictions.
    :param mus: the mean of the individual predictions of an ensemble for a normal distribution.
        The shape should be (n_samples, n_models, n_classes)
    :param sigmas: the standard deviation of the individual probabilities for a normal distribution of an ensemble.
        The shape should be (n_samples, n_models, n_classes)
    :return: total uncertainty, aleatoric uncertainty, epistemic uncertainty
    """
    aleatoric_uncertainty = sigmas.mean(dim=1)
    epistemic_uncertainty = mus.std(dim=1)
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


#### Uncertainty Decomposition Functions used by Evidential Methods ####
def uncertainty_decomposition_evidential_classifier(
    alpha: torch.Tensor,
    classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to compute the uncertainty decomposition based on the evidential classifier.
    :param probabilities: The expected probability of the evidential classifier.
        The shape should be (n_samples, n_classes)
    :param alpha: The alpha parameter of the evidential classifier.
    :param classes: The number of classes.
    :param S: The sum of the alphas
    :return: total uncertainty, aleatoric uncertainty, epistemic uncertainty
    """
    S = alpha.sum(dim=-1, keepdim=True)
    probabilities = alpha / S

    epistemic_uncertainty = (classes / S).view(-1, 1)

    total_uncertainty = (-1 * (probabilities * torch.log2(probabilities))).sum(
        dim=-1, keepdim=True
    )
    aleatoric_uncertainty = -(
        probabilities.mul(digamma(alpha + 1).sub(digamma(S + 1)))
    ).sum(dim=-1, keepdim=True)

    # distributional_uncertainty = (total_uncertainty - aleatoric_uncertainty).clip(0,1)

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def uncertainty_decomposition_evidential_regression(
    beta: torch.Tensor, alpha: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to compute the uncertainty decomposition based on the evidential regression.
    :param beta: The beta parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
    :param alpha: The alpha parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
    :param v: The v parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
    :return: total uncertainty, aleatoric uncertainty, epistemic uncertainty
    """
    aleatoric_uncertainty = beta / (alpha - 1)
    epistemic_uncertainty = beta / (v * (alpha - 1))
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty

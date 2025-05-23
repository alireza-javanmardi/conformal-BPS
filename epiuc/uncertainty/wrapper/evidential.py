from typing import Union

import numpy as np
import torch
from torch.distributions import StudentT
from copy import deepcopy


from epiuc.uncertainty.base import BaseDNN
from epiuc.utils.general import set_model_to_eval
from epiuc.utils.uncertainty_decompositions import (
    uncertainty_decomposition_evidential_classifier,
    uncertainty_decomposition_evidential_regression,
)


class Exp_Layer(torch.nn.Module):
    def __init__(self):
        super(Exp_Layer, self).__init__()

    def forward(self, x):
        return torch.exp(torch.clip(x, -10, 10))


class Evidential_Classifier(BaseDNN):
    """
    This wrapper implements the idea of evidential deep learning, which constructs beliefs on the probability distributions itself in contrast to predictions only one probability distributions.
    Thus natively it captures epistemic uncertainty and aleatoric uncertainty.
    """

    def __init__(
        self,
        base_model: BaseDNN,
        evidence_method: str,
        kl_reg_scaler: float = 1,
        random_state: int = 42,
    ):
        """
        Initializes the Evidential_Classifier object.
        :param base_model: The base uncertainty. This should be a subclass of BaseDNN, and will adapted to an evidential uncertainty.
        For this and extra layer is added to the end of the uncertainty, controlled by the evidence_method.

        :param evidence_method: The method used to calculate the evidence. This should be one of the following:
        "exp", "softplus", "relu".
        :param kl_reg_scaler: The scaling factor for the KL regularization term. This is used to control the
        strength of the regularization, how much the uncertainty is penalized for applying belief to wrong labels.
        :param random_state: The random state to be used for the ensemble. This is used to ensure that the
        ensemble is reproducible.
        """
        base_model = deepcopy(base_model)

        self.evidence_method = evidence_method

        match self.evidence_method:
            case "exp":
                evidence_layer = Exp_Layer()
            case "softplus":
                evidence_layer = torch.nn.Softplus(beta=1.0)
            case "relu":
                evidence_layer = torch.nn.ReLU()
            case _:
                raise ValueError(
                    'The only currenty supported evidential functions: "exp", "softplus", "relu" '
                )

        model = torch.nn.Sequential(*base_model.children(), evidence_layer)

        super().__init__(
            model=model,
            random_state=random_state,
        )

        self.uncertainty_aware = True
        self.name = f"Evidential_Classifier_{evidence_method}_with_{base_model.name}"
        self.kl_reg_scaler = kl_reg_scaler

    def KL_regularization(self, alpha_tilde: torch.Tensor) -> torch.Tensor:
        """
        Kullback Leiber Divergence Regularization for Evidential Networks, controlling how much belief is applied to wrong labels.
        :param alpha_tilde: The alphas of the dirchilet distribution, for the wrong labels.
        :return: The KL divergence between the dirchilet distribution and the uniform distribution.
        """
        n_classes = alpha_tilde.shape[1]

        beta = torch.Tensor(np.ones((1, n_classes))).to(self.device)
        S_alpha = alpha_tilde.sum(dim=1, keepdim=True)
        S_beta = beta.sum(dim=1, keepdim=True)  # K in the paper

        lnB = torch.lgamma(S_alpha) - (torch.lgamma(alpha_tilde)).sum(
            dim=1, keepdim=True
        )
        lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha_tilde)

        kl_divergence = (
            ((alpha_tilde - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True)
            + lnB
            + lnB_uni
        )

        return kl_divergence

    def evidential_mse_loss(
        self,
        global_step: int,
        annealing_step: int,
        target: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evidential MSE Loss for Evidential Networks
        :param global_step: current epoch
        :param annealing_step: number of epochs to anneal the KL regularization
        :param target: true values
        :param alpha: evidence values + 1
        :return: Evidential MSE Loss
        """
        one_hot_targets = torch.nn.functional.one_hot(
            target, num_classes=alpha.shape[1]
        )

        S = alpha.sum(dim=1, keepdim=True)
        evidence = alpha - 1
        m = alpha / S

        A = (one_hot_targets - m).square().sum(dim=1, keepdim=True)
        B = (alpha * (S - alpha) / (S**2 * (S + 1))).sum(dim=1, keepdim=True)

        dirchilet_posterior_loss = A + B

        annealing_coef = min(1.0, global_step / annealing_step)

        alpha_tilde = evidence * (1 - one_hot_targets) + 1  # y_i + (1-y_i) * alpha_i
        kl_regularisation = annealing_coef * self.KL_regularization(alpha_tilde)

        return dirchilet_posterior_loss + self.kl_reg_scaler * kl_regularisation

    def loss_function(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs: dict[str, int]
    ) -> torch.Tensor:

        """
        Computes the loss function for the evidential regression uncertainty.
        :param output: The output of the uncertainty. This should be a tensor of shape (batch_size, n_classes).
        :param target: The target values. This should be a tensor of shape (batch_size,).
        :param kwargs: Additional arguments. This should contain the following keys:
            - iterpoch: The current iteration epoch. This is used to control the annealing of the KL regularization.
            - epoch: The current epoch. This is used to control the annealing of the KL regularization.
            - n_batches: The number of batches in the current epoch. This is used to control the annealing of the KL regularization.
        :return: The loss value. This is a tensor of shape (1,).
        """

        evi_mse_loss = torch.mean(
            self.evidential_mse_loss(
                global_step=kwargs["iterpoch"] + kwargs["epoch"] * kwargs["n_batches"],
                annealing_step=10 * kwargs["n_batches"],
                target=target,
                alpha=output + 1,
            )
        )

        return evi_mse_loss

    def _predict(self, X, raw_output=False) -> tuple[torch.Tensor, torch.Tensor]:

        """
        This method is used to get the predicted probability of the evidential along with the uncertainty.

        :param X: input data
        :param raw_output: If True, return TU,AU, EU of the evidential along with the beliefs, else return mean probability and EU
        :return: The probability and uncertainty of the evidential.
        """
        set_model_to_eval(self.model)

        with torch.no_grad():
            evidence = self(X)

            classes = evidence.shape[1]

            alpha = evidence + 1

            S = alpha.sum(dim=1, keepdim=True)

            probabilities = alpha / S

            (
                total_uncertainty,
                aleatoric_uncertainty,
                epistemic_uncertainty,
            ) = uncertainty_decomposition_evidential_classifier(alpha, classes)

            if raw_output:
                return alpha, torch.stack(
                    [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty],
                    dim=1,
                )
            return probabilities, epistemic_uncertainty


class Evidential_Regression_Final_Layer(torch.nn.Module):
    def __init__(self):
        super(Evidential_Regression_Final_Layer, self).__init__()

    def forward(self, x):
        assert x.shape[1] == 4
        gamma, v, alpha, beta = torch.split(x, 1, dim=-1)

        v = torch.nn.functional.softplus(v)
        alpha = torch.nn.functional.softplus(alpha) + 1
        beta = torch.nn.functional.softplus(beta)

        return torch.cat([gamma, v, alpha, beta], dim=-1)


class Evidential_Regression(BaseDNN):
    """
    This wrapper implements the idea of evidential deep learning, which constructs beliefs on the probability distributions itself in contrast to predictions only one probability distributions.
    Thus natively it captures epistemic uncertainty and aleatoric uncertainty.
    This wrapper is used for regression tasks, and uses the NIG distribution to uncertainty the uncertainty.
    The uncertainty is trained using the NIG loss function, which is a combination of the negative log likelihood and a regularization term.
    The regularization term is used to control the amount of uncertainty in the uncertainty, and is controlled by the nig_lam parameter.
    """

    def __init__(self, base_model: BaseDNN, nig_lam: float = 0, random_state: int = 42):
        """
        Initializes the Evidential_Regression object.
        :param base_model: The base uncertainty. This should be a subclass of BaseDNN, and will adapted to an evidential uncertainty.
        For this and extra layer is added to the end of the uncertainty, controlled by the evidence_method.
        :param: nig_lam: The scaling factor for the NIG regularization term.
        :param: random_state: The random state to use for initializing the evidential uncertainty.
        """

        model = deepcopy(base_model.model)

        model = self._build_evidential_reg_model(model)

        super(Evidential_Regression, self).__init__(
            model=model,
            random_state=random_state,
        )

        # Class params
        self.name = f"Evidential_Regression_on_{base_model.name}"
        self.nig_lam: float = nig_lam
        self.uncertainty_aware = True

        # Parameters for regularization
        self.epsilon: float = 1e-2
        self.maxi_rate: float = 1e-4

    def _build_evidential_reg_model(self, model):
        """
        This method is used to build the evidential regression uncertainty.

        For this it searches for the last linear layer in the uncertainty, and then adds an extra layer to the end of the uncertainty.
        :param model: The base uncertainty. This should be a subclass of BaseDNN, and will adapted to an evidential uncertainty.
        :return: The evidential regression uncertainty.
        """
        model_children = list(model.children())[::-1]
        idx_remove_layers = 0
        for i, layer in enumerate(model_children):
            if isinstance(layer, torch.nn.Linear):
                idx_remove_layers = i
                break
        head_in_features = layer.in_features
        head = torch.nn.Sequential(
            torch.nn.Linear(in_features=head_in_features, out_features=4, bias=True),
            Evidential_Regression_Final_Layer(),
        )
        # Build a new uncertainty, consiting of the base uncertainty without the last layers and the new head
        model = torch.nn.Sequential(
            *(model_children[::-1])[: len(model_children) - (idx_remove_layers + 1)],
            head,
        )
        return model

    def nig_nll_loss(
        self,
        y: torch.Tensor,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log likelihood loss for the NIG distribution.
        :param y: The target values. This should be a tensor of shape (batch_size,).
        :param gamma: The gamma parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param v: The v parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param alpha: The alpha parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param beta: The beta parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :return: The negative log likelihood loss. This is a tensor of shape (batch_size,).
        """
        student_var = beta * (1.0 + v) / (v * alpha)

        dist = StudentT(loc=gamma, scale=student_var, df=2 * alpha)

        nll = -1.0 * dist.log_prob(y)

        return nll.mean()

    def nig_regularization(
        self, y: torch.Tensor, gamma: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Regularization term for the NIG distribution.
        :param y: The target values. This should be a tensor of shape (batch_size,).
        :param gamma: The gamma parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param v: The v parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param alpha: The alpha parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :param beta: The beta parameter of the NIG distribution. This should be a tensor of shape (batch_size,).
        :return: The regularization term. This is a tensor of shape (batch_size,).
        """
        error = torch.abs(y - gamma)
        evi = 2 * v + alpha

        return (error * evi).mean()

    def loss_function(self, output, target, **kwargs):
        """
        Computes the loss function for the evidential regression uncertainty.
        This is a mixture of the negative log likelihood and a regularization term.
        :param output: The output of the uncertainty. This should be a tensor of shape (batch_size, 4).
        :param target: The target values. This should be a tensor of shape (batch_size,).
        :param kwargs: Additional arguments, but not used.
        """
        gamma, v, alpha, beta = torch.split(output, 1, dim=-1)

        loss_nll = self.nig_nll_loss(target, gamma, v, alpha, beta)
        loss_reg = self.nig_regularization(target, gamma, v, alpha)

        # additional -epsilon in regularization loss
        loss = loss_nll + self.nig_lam * (loss_reg - self.epsilon)

        return loss

    def _predict(
        self, X: np.ndarray | torch.Tensor, raw_output: bool = False
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        This method is used to get the prediction of the evidential along with the uncertainty.
        :param X: input data
        :param raw_output: If True, return TU,AU, EU of the evidential along with the NIG parameters, else return mean probability and EU
        :return: The mean prediction and uncertainty of the evidential regression uncertainty.
        """
        set_model_to_eval(self.model)

        with torch.no_grad():
            mu, v, alpha, beta = torch.split(self(X), 1, dim=-1)

            (
                total_uncertainty,
                aleatoric_uncertainty,
                epistemic_uncertainty,
            ) = uncertainty_decomposition_evidential_regression(beta, alpha, v)

            if raw_output:
                return (
                    torch.stack([mu, v, alpha, beta], dim=1),
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

                return mu, epistemic_uncertainty

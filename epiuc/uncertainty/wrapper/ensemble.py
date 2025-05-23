from copy import deepcopy
from typing import Union

import numpy as np

from epiuc.uncertainty.base import BaseEnsembleDNN, BaseDNN
import torch

from epiuc.utils.general import validate_predict_input, set_model_to_eval
from epiuc.utils.uncertainty_decompositions import (
    uncertainty_decomposition_entropy,
    uncertainty_decomposition_mus_sigmas,
)


class Ensemble_Classifier(BaseEnsembleDNN):
    """
    This wrapper implements the idea of caputring the (epistemic) uncertainty of a uncertainty by an ensemble of models.

    The ensemble is created by training multiple models on the same data, but with different initializations and/or
    different subsets of the data, whether the base_model consists of and list of models or a single uncertainty.

    The idea is that the models will learn different representations of the data, and
    thus will have different predictions. By averaging the predictions of the models, we can get a more robust
    prediction, and also capture the uncertainty of the uncertainty.
    """

    def __init__(
        self,
        base_model: BaseDNN | list[BaseDNN],
        n_models: int,
        random_state: int = 42,
    ):
        """
        Initializes the Ensemble_Classification object.
        :param base_model: The base uncertainty to be used for the ensemble. This should be a subclass of BaseDNN.
        :param n_models: The number of models to be used in the ensemble.
        :param random_state: The random state to be used for the ensemble. This is used to ensure that the
        ensemble is reproducible.
        """
        super(Ensemble_Classifier, self).__init__(
            n_models=n_models,
            model=base_model,
            random_state=random_state,
        )
        self.name: str = "Ensemble_Classification"
        self.uncertainty_aware: bool = True

    def _predict(
        self, X: torch.Tensor, raw_output=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method is used to get the predicted probability of the ensemble along with the uncertainty.
        :param X: input data
        :param raw_output: If True, return TU,AU, EU of the ensemble along with the probabilities from the individual models, else return mean probability and EU
        :return: The probability and uncertainty of the ensemble.
        """
        X = validate_predict_input(X)

        set_model_to_eval(self.ensemble)

        res = []
        for model in self.ensemble:
            model.eval()
            res.append(model._predict(X))

        probs = torch.stack(res, dim=1)

        (
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = uncertainty_decomposition_entropy(probs)

        # Break point if nan is detected
        if (
            torch.isnan(total_uncertainty).any()
            or torch.isnan(aleatoric_uncertainty).any()
        ):
            raise ValueError(
                "NaN detected in the uncertainty decomposition. Please check your input data."
            )

        if raw_output:
            return probs, torch.stack(
                [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty], dim=1
            )
        return probs.mean(dim=1), epistemic_uncertainty.view(-1, 1)


class Ensemble_Regression(BaseEnsembleDNN):

    """
    This wrapper implements the idea of capturing the (epistemic) uncertainty of a uncertainty by an ensemble of models.

    The ensemble is created by training multiple models on the same data, but with different initializations and/or
    different subsets of the data, whether the base_model consists of and list of models or a single uncertainty.

    The idea is that the models will learn different representations of the data, and
    thus will have different predictions. By averaging the predictions of the models, we can get a more robust
    prediction, and also capture the uncertainty of the uncertainty.
    """

    def __init__(
        self,
        base_model: BaseDNN | list[BaseDNN],
        n_models: int,
        random_state: int = 42,
    ):
        """
        Initializes the Ensemble_Regression object.
        :param base_model: The base uncertainty to be used for the ensemble. This should be a subclass of BaseDNN and output both mean and sigma of a distribution.
        :param n_models: The number of models to be used in the ensemble.
        :param random_state: The random state to be used for the ensemble. This is used to ensure that the
            ensemble is reproducible.
        """
        super(Ensemble_Regression, self).__init__(
            n_models=n_models,
            model=deepcopy(base_model),
            random_state=random_state,
        )

        self.sigma = base_model.sigma

        self.name = "Ensemble_Regression"
        self.uncertainty_aware = True

    def _predict(
        self,
        X: np.ndarray | torch.Tensor,
        raw_output: bool = False,
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        This method is used to get the mean prediction of the ensemble along with the uncertainty.
        :param X: input data
        :param raw_output: If True, return TU,AU, EU of the ensemble along with the predictions of the individual models, else return mean prediction and EU
        :return:
        """
        X = validate_predict_input(X)

        set_model_to_eval(self.ensemble)

        output = self.forward(X)
        mus, sigmas = output.split(1, dim=-1)

        mean_mus = mus.mean(dim=1)

        (
            total_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = uncertainty_decomposition_mus_sigmas(mus, sigmas)

        if raw_output:
            return torch.stack([mus, sigmas], dim=2), torch.stack(
                [
                    total_uncertainty,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                ],
                dim=1,
            )
        else:
            return mean_mus, epistemic_uncertainty

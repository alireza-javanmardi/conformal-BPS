import copy
from typing import Optional
from warnings import warn

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import DataLoader, random_split

from epiuc.config import (
    BACKEND,
    DATALOADER_CONFIGS,
    SCHEDULER_CONFIGS,
    OPTIMIZER_CONFIGS,
)
from epiuc.uncertainty.base import (
    BaseDNN,
    BaseEnsembleDNN,
    _convert_to_dataloader,
)
from epiuc.uncertainty.regression import MLP_Quantile
from epiuc.conformal_prediction.eval_metrics import vio_classes, WSC, covGap
from epiuc.conformal_prediction.non_conform_func import (
    cqr_score,
    aps_non_conformity_scores,
    raps_non_conformity_scores,
    cqr_r_score,
    uacqr_s_score,
)
from epiuc.utils.general import (
    to_numpy,
    calculate_quantile,
    gather_results,
    find_device,
    validate_predict_input,
)


###################################
# Classification Conformal Methods #
###################################


def create_one_hot_encoding(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Create a one-hot encoding of the class indices.
    :param indices: Class indices to be one-hot encoded.
    :param num_classes: Number of classes.
    :return: One-hot encoded array.
    """
    indices = np.asarray(indices)  # Ensure indices is a NumPy array
    if np.any(indices >= num_classes) or np.any(indices < 0):
        raise ValueError("Class indices must be between 0 and num_classes-1.")

    # Create the one-hot encoded array
    one_hot_array = np.zeros((indices.size, num_classes), dtype=np.int64)
    one_hot_array[np.arange(indices.size), indices.ravel()] = 1

    # Reshape to match the input shape, adding the num_classes dimension
    return one_hot_array.reshape(*indices.shape, num_classes)


class MondrianAPS:
    """
    Mondrian Adaptive Prediction Sets (MAPS) for classification building bins based on (epistemic) uncertainty
    """

    def __init__(
        self, model: BaseDNN, n_classes: int, n_bins: int, randomness: bool = True
    ):
        """
        Initialize the MAPS) for classification.
        :param model: classification uncertainty
        :param n_classes: number of classes
        :param n_bins: number of bins
        :param randomness: Whether to use randomized APS
        """
        self.model = copy.deepcopy(model)
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.randomness = randomness

        # Main parameters that need calibration
        self.bins = None
        self.bins_non_conformity_scores = {
            bin_id: np.array([-1]) for bin_id in range(1, n_bins + 1)
        }
        self.bins_quantile = {bin_id: -1.0 for bin_id in range(1, n_bins + 1)}

    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor | DataLoader,
        y_cal: Optional[np.ndarray | torch.Tensor] = None,
        alpha: float = 0.1,
    ) -> None:
        """
        Calibrate the models quantile based on the calibration set
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        calloader = _convert_to_dataloader(X_cal, y_cal)

        self.model.eval()

        probabilities, uncertainty, _, y = gather_results(self.model, calloader)

        # Create n_bins based on uncertainty and group probabilities and y accordingly
        uncertainty = uncertainty.flatten()
        self.bins = np.linspace(0, np.max(uncertainty), self.n_bins)

        # get index of the bins the uncertainty belongs to
        bin_idx = np.digitize(uncertainty, self.bins)

        for bin_id in np.unique(bin_idx):
            # filter the data for each bin
            bin_prob = probabilities[bin_idx == bin_id]
            bin_y = y[bin_idx == bin_id]

            bin_aps_scores = aps_non_conformity_scores(
                bin_prob, bin_y, randomized=self.randomness
            )

            # build quantile
            self.bins_non_conformity_scores[bin_id] = bin_aps_scores
            self.bins_quantile[bin_id] = calculate_quantile(bin_aps_scores, alpha=alpha)

        self.alpha = alpha

    def _build_pred_set(
        self, probabilities: np.ndarray, uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Build the one-hot encoded prediction set based on the quantile
        :param probabilities: probabilities of the uncertainty
        :param uncertainty: (epistemic) uncertainty of the uncertainty
        :return: one-hot encoded prediction set
        """
        uncertainty = uncertainty.flatten()
        # get index of the bins the uncertainty belongs to
        bin_idx = np.digitize(uncertainty, self.bins)
        set_predictions = np.zeros((len(probabilities), self.n_classes), dtype=bool)

        for bin_id in np.unique(bin_idx):
            # filter the probability for the bin
            bin_prob = probabilities[bin_idx == bin_id]

            set_predictions[bin_idx == bin_id] = (
                aps_non_conformity_scores(bin_prob, randomized=self.randomness)
                <= self.bins_quantile[bin_id]
            )
        return set_predictions

    def predict(self, X_test: np.ndarray | torch.Tensor | DataLoader) -> np.ndarray:
        """
        Predict the set of classes for the tests data.
        :param X_test:
        :return: one-hot encoded prediction set
        """

        self.model.eval()
        probabilities, uncertainty = self.model.predict(X_test, raw_output=False)
        probabilities = probabilities.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

        set_predictions = self._build_pred_set(probabilities, uncertainty)

        return set_predictions

    def evaluate(
        self,
        X_eval: np.ndarray | torch.Tensor | DataLoader,
        y_eval: Optional[np.ndarray | torch.Tensor] = None,
    ) -> tuple[float, float, int, float, float]:
        """
        Evaluate the uncertainty based on the evaluation set.
        :param X_eval: evaluation data
        :param y_eval: evaluation targets
        :return: coverage, avg_set_size, violated_classes, coverage_gap, wsc
        """
        # Deactivate shuffle for the dataloader
        calloader = _convert_to_dataloader(X_eval, y_eval, shuffle=False)

        probabilities, uncertainty, features, y = gather_results(self.model, calloader)

        set_predictions = self._build_pred_set(probabilities, uncertainty)

        set_size = np.sum(set_predictions, axis=1)
        covered = set_predictions[
            create_one_hot_encoding(y, self.n_classes).astype(bool)
        ]
        coverage = np.sum(covered) / covered.shape[0]

        avg_set_size = np.mean(set_size)

        violated_classes = vio_classes(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        coverage_gap = covGap(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        wsc = WSC(features, set_predictions, y)

        return coverage, avg_set_size, violated_classes, coverage_gap, wsc


class APS:
    """
    Adaptive Prediction Sets (APS) for classification
    """

    def __init__(
        self,
        model: BaseDNN,
        n_classes: int,
        randomness: bool = True,
        epiuc_t: float = 0.5,
    ):
        """
        Initialize the Adaptive Prediction Sets (APS) for classification.
        :param model: classification uncertainty
        :param n_classes: number of classes
        :param randomness: Whether to use randomized APS
        :param epiuc_t: multiplicative factor for incorporating (epistemic) uncertainty
        """
        self.model = copy.deepcopy(model)
        self.n_classes = n_classes
        self.epiuc_t = epiuc_t
        self.randomness = randomness

        # data to be set in calibration
        self.quantile = None
        self.non_conform_scores = None
        self.alpha = None

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        dataset_name: str | None = None,
        n_epochs: int = 50,
        batch_size: int = DATALOADER_CONFIGS["batch_size"],
        optimizer_params: dict[str, int] = OPTIMIZER_CONFIGS["adam"],
        scheduler_params: dict[str, int] = SCHEDULER_CONFIGS["step"],
    ):
        """
        Function to fit the deep learning uncertainty
        :param X: training data
        :param y: training targets
        :param dataset_name: name of the dataset to store the weights of the uncertainty, if set
        :param n_epochs: number of epochs
        :param batch_size: batch size for the dataloader to be used, if torch.Tensor or numpy array is given
        :param optimizer_params: dictionary containing the optimizer and its parameters in the field "optimizer" and "params" respectively. See `config.py` for more information.
        :param scheduler_params: dictionary containing the scheduler and its parameters in the field "scheduler" and "params" respectively. See `config.py` for more information.
        :return: trained uncertainty
        """

        return self.model.fit(
            X,
            y,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
        )

    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor | DataLoader,
        y_cal: Optional[np.ndarray | torch.Tensor] = None,
        alpha=0.1,
    ) -> None:
        """
        Calibrate the models quantile based on the calibration set
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        calloader = _convert_to_dataloader(X_cal, y_cal)

        self.model.eval()

        probabilities, uncertainty, _, y = gather_results(self.model, calloader)

        # Make sure the values are between 0 and 1 if inducing (epistemic) uncertainty
        self.non_conform_scores = (
            aps_non_conformity_scores(probabilities, y, randomized=self.randomness)
            - self.epiuc_t * uncertainty.flatten()
        ).clip(0, 1)

        self.alpha = alpha
        self.quantile = calculate_quantile(self.non_conform_scores, alpha=alpha)

    def plot_non_conformity_scores(self):
        """
        Plot the non-conformity scores for the calibration set
        :return:
        """
        import matplotlib.pyplot as plt

        plt.hist(self.non_conform_scores, color="b", bins=50)
        plt.axvline(x=self.quantile, color="r", linestyle="--", label="Quantile")
        plt.axvline(x=1 - self.alpha, color="g", linestyle="--", label="1-alpha")
        plt.xlabel("Non-conformity scores")
        plt.ylabel("Frequency")
        plt.title(f"Non-conformity scores Distribution for EAPS_t={self.epiuc_t}")
        plt.legend()
        plt.savefig(f"EAPS_t{self.epiuc_t}.pdf")
        plt.show()

    def _build_pred_set(
        self, probabilities: np.ndarray, uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Build the one-hot encoded prediction set based on the quantile
        :param probabilities: probabilities of the uncertainty
        :param uncertainty: (epistemic) uncertainty of the uncertainty
        :return: one-hot encoded prediction set
        """
        scores = (
            aps_non_conformity_scores(probabilities, randomized=self.randomness)
            - self.epiuc_t * uncertainty
        ).clip(0, 1)

        pred_set = scores <= self.quantile

        return pred_set

    def predict(self, X_test: np.ndarray | torch.Tensor | DataLoader) -> np.ndarray:
        """
        Predict the set of classes for the tests data.
        :param X_test: tests data
        :return: one-hot encoded prediction sets
        """
        self.model.eval()

        probabilities, uncertainty = self.model.predict(X_test)
        probabilities = probabilities.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

        set_predictions = self._build_pred_set(probabilities, uncertainty)

        return set_predictions

    def evaluate(
        self,
        X_eval: np.ndarray | torch.Tensor | DataLoader,
        y_eval: Optional[np.ndarray | torch.Tensor] = None,
    ) -> tuple[float, float, int, float, float]:
        """
        Evaluate the uncertainty based on the evaluation set.
        :param X_eval: evaluation data
        :param y_eval: evaluation targets
        :return: coverage, avg_set_size, violated_classes, coverage_gap, wsc
        """
        calloader = _convert_to_dataloader(X_eval, y_eval)

        probabilities, uncertainty, features, y = gather_results(self.model, calloader)

        set_predictions = self._build_pred_set(probabilities, uncertainty)

        set_size = np.sum(set_predictions, axis=1)
        covered = set_predictions[
            create_one_hot_encoding(y, self.n_classes).astype(bool)
        ]
        coverage = np.sum(covered) / covered.shape[0]

        avg_set_size = np.mean(set_size)

        violated_classes = vio_classes(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        coverage_gap = covGap(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        wsc = WSC(features, set_predictions, y)

        return coverage, avg_set_size, violated_classes, coverage_gap, wsc


class RAPS(APS):
    """
    Regularized Adaptive Prediction Sets (RAPS) for classification
    """

    def __init__(
        self,
        model: BaseDNN,
        n_classes: int,
        randomness: bool = True,
        epiuc_t: float = 0.5,
    ):
        """
        Initialize the Regularized Adaptive Prediction Sets (RAPS) for classification.
        :param model: classifier uncertainty
        :param n_classes: number of classes
        :param randomness: Whether to use randomized APS
        :param epiuc_t: multiplicative factor for incorporating (epistemic) uncertainty
        """
        super().__init__(model, n_classes, randomness, epiuc_t)

        # These are the default values for the RAPS method.
        self.lam = 0.001
        self.k_reg = 5

    def _find_best_k_star(
        self, probabilities: np.ndarray, y: np.ndarray, alpha
    ) -> float:
        _, indices = (
            np.sort(probabilities, axis=-1)[:, ::-1],
            np.argsort(probabilities, axis=1)[:, ::-1],
        )
        relevant_index = np.where(indices == y[:, None])
        k_star = calculate_quantile(relevant_index[1] + 1, 1 - alpha)
        return k_star

    def _find_best_lambda(self, dataloader: DataLoader, alpha: float) -> float:

        # Save optimal values
        best_set_size = np.inf
        best_lamdba = 0.001

        # Extensive search for lambda
        for lam in [0.001, 0.01, 0.1, 0.5]:
            # Act as if this lambda would now be choosen

            self.lam = lam
            probabilities, uncertainty, _, y = gather_results(self.model, dataloader)

            self.non_conform_scores = raps_non_conformity_scores(
                probabilities, y, self.lam, self.k_reg
            )
            self.quantile = calculate_quantile(self.non_conform_scores, alpha=alpha)
            set_prediction = self.predict(dataloader)
            # calculate mean set size
            set_size = np.mean(np.sum(set_prediction, axis=1))

            # smaller set_size is better
            if set_size < best_set_size:
                best_set_size = set_size
                best_lamdba = lam

        return best_lamdba

    def _build_pred_set(
        self, probabilities: np.ndarray, uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Build the one-hot encoded prediction set based on the quantile
        :param probabilities: probabilities of the uncertainty
        :param uncertainty: (epistemic) uncertainty of the uncertainty
        :return: one-hot encoded prediction set
        """
        scores = (
            raps_non_conformity_scores(
                probabilities, None, self.lam, self.k_reg, randomized=self.randomness
            )
            - self.epiuc_t * uncertainty
        ).clip(0, 1)

        pred_set = scores <= self.quantile

        return pred_set

    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor | DataLoader,
        y_cal: np.ndarray | torch.Tensor = None,
        alpha=0.1,
    ):
        """
        Calibrate the models quantile based on the calibration set
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        self.model.eval()

        dataloader = _convert_to_dataloader(X_cal, y_cal)

        # Split dataloader into two parts
        cal_size = int(0.8 * len(dataloader.dataset))
        tune_size = len(dataloader.dataset) - cal_size

        # Split the dataset
        train_dataset, val_dataset = random_split(
            dataloader.dataset, [cal_size, tune_size]
        )

        # Create DataLoaders for each split
        cal_loader = DataLoader(train_dataset, **DATALOADER_CONFIGS)
        tune_loader = DataLoader(val_dataset, **DATALOADER_CONFIGS)

        probabilities_tune, _, _, y_tune = gather_results(self.model, tune_loader)

        self.k_reg = self._find_best_k_star(probabilities_tune, y_tune, alpha)

        self.lam = self._find_best_lambda(tune_loader, alpha)

        probabilities_cal, _, _, y_cal = gather_results(self.model, cal_loader)

        self.non_conform_scores = raps_non_conformity_scores(
            probabilities_cal, y_cal, self.lam, self.k_reg
        )
        self.quantile = calculate_quantile(self.non_conform_scores, alpha=alpha)

    def predict(self, X_test: np.ndarray | torch.Tensor | DataLoader) -> np.ndarray:
        """
        Predict the set of classes for the tests data.
        :param X_test: tests data
        :return: one-hot encoded sets
        """
        self.model.eval()

        probabilities, uncertainty = self.model.predict(X_test)
        probabilities = probabilities.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

        return self._build_pred_set(probabilities, uncertainty)


class APS_Belief(APS):
    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor,
        y_cal: Optional[np.ndarray | torch.Tensor] = None,
        alpha=0.1,
    ) -> None:
        """
        Calibrate the models quantile based on the calibration set
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """

        self.model.eval()

        calloader = _convert_to_dataloader(X_cal, y_cal)

        beliefs, features, y = self.gather_results(calloader)

        # Make sure the values are between 0 and 1 if inducing (epistemic) uncertainty
        self.non_conform_scores = (
            aps_non_conformity_scores(beliefs, y, randomized=self.randomness)
        ).clip(0, 1)

        self.alpha = alpha
        self.quantile = calculate_quantile(self.non_conform_scores, alpha=alpha)

    def _build_pred_set(self, beliefs: np.ndarray) -> np.ndarray:
        """
        Build the one-hot encoded prediction set based on the quantile
        :param probabilities: probabilities of the uncertainty
        :param uncertainty: (epistemic) uncertainty of the uncertainty
        :return: one-hot encoded prediction set
        """

        scores = (aps_non_conformity_scores(beliefs, randomized=self.randomness)).clip(
            0, 1
        )

        pred_set = scores <= self.quantile

        return pred_set

    def gather_results(self, dataloader):
        beliefs = None
        features = None
        y = None
        i = 0
        batch_size = dataloader.batch_size
        for data in iter(dataloader):
            inputs, labels = data[0], data[1]

            # Every data instance is an input + label pair
            current_batch_size, remain = inputs.shape[0], inputs.shape[1:]

            # Pad the input to have same batch size
            inputs = torch.cat(
                [inputs, torch.zeros(batch_size - current_batch_size, *remain)],
                dim=0,
            )

            # move to device
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)

            alphas, _ = self.model.predict(inputs, raw_output=True)
            belief = (alphas - 1).div(alphas.sum(dim=-1, keepdim=True))
            belief = belief.cpu().numpy()

            # Reset Padding
            inputs = inputs[:current_batch_size]
            if i == 0:
                beliefs = belief

                beliefs = beliefs[:current_batch_size]

                features = (
                    inputs.to(torch.device("cpu")).numpy().reshape(inputs.shape[0], -1)
                )
                y = labels.to(torch.device("cpu")).numpy()
            else:

                belief = belief[:current_batch_size]

                beliefs = np.vstack([beliefs, belief])

                features = np.vstack(
                    [
                        features,
                        inputs.to(torch.device("cpu"))
                        .numpy()
                        .reshape(inputs.shape[0], -1),
                    ]
                )
                y = np.hstack([y, labels.to(torch.device("cpu")).numpy()])
            i += 1

        return beliefs, features, y

    def evaluate(
        self,
        X_eval: np.ndarray | torch.Tensor,
        y_eval: Optional[np.ndarray | torch.Tensor] = None,
    ) -> tuple[float, float, int, float, float]:
        """
        Evaluate the uncertainty based on the evaluation set.
        :param X_eval: evaluation data
        :param y_eval: evaluation targets
        :return: coverage, avg_set_size, violated_classes, coverage_gap, wsc
        """
        calloader = _convert_to_dataloader(X_eval, y_eval)

        beliefs, features, y = self.gather_results(calloader)

        set_predictions = self._build_pred_set(beliefs)

        set_size = np.sum(set_predictions, axis=1)
        covered = set_predictions[
            create_one_hot_encoding(y, self.n_classes).astype(bool)
        ]
        coverage = np.sum(covered) / covered.shape[0]

        avg_set_size = np.mean(set_size)

        violated_classes = vio_classes(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        coverage_gap = covGap(
            set_predictions, y, alpha=self.alpha, num_classes=self.n_classes
        )

        wsc = WSC(features, set_predictions, y)

        return coverage, avg_set_size, violated_classes, coverage_gap, wsc

    def predict(self, X_test: np.ndarray | torch.Tensor | DataLoader) -> np.ndarray:
        """
        Predict the set of classes for the tests data.
        :param X_test: tests data
        :return: one-hot encoded prediction sets
        """
        self.model.eval()

        alphas, _ = self.model.predict(X_test, raw_output=True)
        beliefs = (alphas - 1) / np.sum(alphas, axis=1, keepdims=True)
        beliefs = beliefs.cpu().numpy()
        set_predictions = self._build_pred_set(beliefs)

        return set_predictions


###################################
# Regression Conformal Methods ####
###################################
class CQR:
    """
    Conformalized Quantile Regression
    """

    def __init__(
        self,
        quantile_model: BaseDNN,
        random_state=42,
    ):
        self.model = copy.deepcopy(quantile_model)
        self.threshold = None

    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor,
        y_cal: np.ndarray | torch.Tensor,
        alpha: float = 0.1,
    ) -> None:
        """
        Calibrate the non_conformity threshold
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        self.model.eval()

        predictions_cal = self.model.predict(X_cal)
        predictions_cal = predictions_cal.cpu().numpy()

        if type(y_cal) == torch.Tensor:
            y_cal = to_numpy(y_cal)

        non_conform_scores = cqr_score(intervals=predictions_cal, y_true=y_cal)

        print(non_conform_scores.shape, len(y_cal))

        # Get quantile of non-conform scores
        self.threshold = calculate_quantile(non_conform_scores, alpha=alpha)

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor | torch.utils.data.DataLoader,
        Y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray | torch.Tensor | torch.utils.data.DataLoader] = None,
        Y_val: Optional[np.ndarray] = None,
        dataset_name: Optional[str] = None,
        n_epochs: int = 50,
        batch_size: int = DATALOADER_CONFIGS["batch_size"],
        optimizer_params: dict[str, int] = OPTIMIZER_CONFIGS["adam"],
        scheduler_params: dict[str, int] = SCHEDULER_CONFIGS["step"],
    ):
        """
        Function to fit the deep learning uncertainty given at __init__.
        This function will also save the uncertainty to disk if a dataset name is provided.
        :param X_train: features or complete data loader
        :param Y_train: labels or None
        :param X_val: features or complete data loader
        :param Y_val: labels or None
        :param dataset_name: name of the dataset to save the uncertainty. If None the uncertainty will not be saved to an folder, otherwise it will be saved to the path models/dataset_name/
        :param n_epochs: number of epochs
        :param batch_size: batch size for the dataloader to be used, if torch.Tensor or numpy array is given
        :param optimizer_params: dictionary containing the optimizer and its parameters in the field "optimizer" and "params" respectively. See `config.py` for more information.
        :param scheduler_params: dictionary containing the scheduler and its parameters in the field "scheduler" and "params" respectively. See `config.py` for more information.
        :return: self
        """
        return self.model.fit(
            X_train,
            Y_train,
            X_val,
            Y_val,
            dataset_name,
            n_epochs,
            batch_size,
            optimizer_params,
            scheduler_params,
        )

    @torch.no_grad()
    def _predict(self, X: torch.Tensor | np.ndarray, raw_outputs=False):
        """
        Predict conformalized intervals
        :param X: tests data
        :param raw_outputs: If True, outputs non-conformalized intervals
        :return: conformalized intervals, or non-conformalized intervals and threshold if raw_outputs is True
        """

        predictions_test = self.model._predict(X, raw_outputs=raw_outputs)

        cqr_interval = np.vstack(
            [
                predictions_test[:, 0] - self.threshold,
                predictions_test[:, 1] + self.threshold,
            ]
        ).T
        cqr_interval = cqr_interval

        if raw_outputs:
            return predictions_test, self.threshold

        return torch.Tensor(cqr_interval)

    @torch.no_grad()
    def predict(self, X: torch.Tensor | np.ndarray | DataLoader, raw_outputs=False):
        self.model.eval()
        if self.threshold is None:
            raise ValueError(
                "Model not calibrated. Please call the calibrate function before predicting."
            )
        if type(X) == DataLoader:
            output = [
                self._predict(inputs, raw_outputs=raw_outputs) for inputs, _ in iter(X)
            ]
            return torch.cat(output)
        else:
            X = validate_predict_input(X, self.model.device)

            return self._predict(X, raw_outputs=raw_outputs)


class CQR_r(CQR):
    """
    Conformalized Quantile Regression incorporating the width of prediction intervals into non-conformity scores
    """

    def calibrate(
        self,
        X_cal: np.ndarray | torch.Tensor,
        y_cal: np.ndarray | torch.Tensor,
        alpha: float = 0.1,
    ) -> None:
        """
        Calibrate the non_conformity threshold
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """

        predictions_cal = self.model._predict(X_cal)
        predictions_cal = predictions_cal.cpu().numpy()

        if type(y_cal) == torch.Tensor:
            y_cal = to_numpy(y_cal)

        non_conform_scores = cqr_r_score(predictions_cal, y_cal)

        # Get Quantile of non-conform scores
        print(non_conform_scores.shape, len(y_cal))
        self.threshold = calculate_quantile(non_conform_scores, alpha=alpha)

    @torch.no_grad()
    def _predict(self, X, raw_outputs=False):
        """
        Predict conformalized intervals with width scaling
        :param X: tests data
        :param raw_outputs: If True, outputs non-conformalized intervals
        :return: conformalized intervals, or non-conformalized intervals and threshold if raw_outputs is True
        """

        predictions_test = self.model._predict(X, raw_outputs=raw_outputs)

        if self.threshold is None:
            return predictions_test

        width = predictions_test[:, 1] - predictions_test[:, 0]

        cqr_interval = np.vstack(
            [
                predictions_test[:, 0] - self.threshold * width,
                predictions_test[:, 1] + self.threshold * width,
            ]
        ).T
        if raw_outputs:
            return predictions_test, self.threshold
        return torch.Tensor(cqr_interval)


class DCP(BaseEnsembleDNN):
    """
    Distributional Conformal Prediction.

    Rather than conformalizing the intervals, a quantile uncertainty is picked that yields the desired coverage.
    """

    def __init__(self, input_shape, n_layers, num_neurons, dropout_rate):
        quantiles = [(i, 1 - i) for i in np.arange(0.05, 0.5, 0.05)]
        base_models = [
            MLP_Quantile(
                input_shape=input_shape,
                n_layers=n_layers,
                num_neurons=num_neurons,
                dropout_rate=dropout_rate,
                quantiles=list(pairs),
            )
            for pairs in quantiles
        ]
        self.quantile = None
        super().__init__(n_models=len(base_models), model=base_models)

    def calibrate(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, alpha=0.1
    ):
        """
        Calibrate the non_conformity threshold
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """

        [model.eval() for model in self.ensemble]
        if type(y) == torch.Tensor:
            y = to_numpy(y)

        y = y.flatten()

        min_val_coverage = np.inf
        for i, model in enumerate(self.ensemble):
            model.eval()
            with torch.no_grad():
                predictions_cal = model.predict(X)
            predictions_cal = to_numpy(predictions_cal)

            convered_y = np.logical_and(
                predictions_cal[:, 0] <= y, y <= predictions_cal[:, 1]
            )
            if np.sum(convered_y) < min_val_coverage and np.sum(convered_y) > (
                1 - alpha
            ) * (len(X) + 1):
                min_val_coverage = np.sum(convered_y)
                self.quantile = i

        if self.quantile is None:
            warn(
                "No quantile uncertainty satisfies the coverage requirement. This may due to overfitting."
                "For the prediction the uncertainty with the highest coverage will be used."
            )

    def _predict(self, X, raw_outputs=False):
        [model.eval() for model in self.ensemble]

        if self.quantile is None:
            cqr_interval = self.ensemble[0].predict(X)
        else:
            cqr_interval = self.ensemble[self.quantile].predict(X)

        if raw_outputs:
            return cqr_interval, self.quantile
        return cqr_interval


class UACQR:
    """
    Uncertainty Aware Conformalized Quantile Regression.

    This clase should not be used directly, but rather its sub_classes UACQR_S & UACQR_P.
    These methods indirectly incorporate epistemic uncertainty into the calibration of the threshold.
    This is achieved by saving the individual models states after each epoch.
    """

    def __init__(self, quantile_model: BaseDNN):
        self.model = quantile_model

        self.model.init_optimizer_scheduler(
            OPTIMIZER_CONFIGS["adam"], SCHEDULER_CONFIGS["step"]
        )

        self.epoch_tracking: bool = True
        self.saved_models: list[BaseDNN] = []

        self.uncertainty_aware: bool = True

    def fit(
        self,
        X: ndarray | DataLoader,
        Y: ndarray | None = None,
        dataset_name: str | None = None,
        n_epochs: int = 50,
        batch_size: int = DATALOADER_CONFIGS["batch_size"],
    ):
        """
        Function to fit the deep learning uncertainty
        This function will also save the uncertainty to disk if a dataset name is provided.
        :param X: features or complete data loader
        :param Y: labels or None
        :param dataset_name: name of the dataset to save the uncertainty. If None the uncertainty will not be saved to an folder, otherwise it will be saved to the path models/dataset_name/
        :param n_epochs: number of epochs
        :param batch_size: batch size for the dataloader to be used, if torch.Tensor or numpy array is given
        :return: self
        """
        dataloader = _convert_to_dataloader(X, Y, batch_size=batch_size)

        self.fit_trainloader(dataloader, dataset_name, n_epochs)

    def fit_trainloader(self, trainloader, dataset_name, n_epochs):
        """
        Function to fit the deep learning uncertainty with a trainloader object
        :param trainloader: training data loader
        :param n_epochs: number of epochs
        :return: Saves the uncertainty to disk and returns None
        """
        # Setup
        n_batches = len(trainloader)

        # Set the uncertainty to training mode
        self.model.set_device(find_device())
        self.model.compile(backend=BACKEND)

        for epoch in range(n_epochs):
            current_epoch = epoch
            print("EPOCH {}:".format(current_epoch))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)

            avg_loss = 0

            for i, data in enumerate(trainloader):
                # Every data instance is an input + label pair
                inputs, labels = data[0].to(self.model.device), data[1].to(
                    self.model.device
                )

                avg_loss += self.model.train_step(inputs, labels)

            # Update the learning rate
            self.model.scheduler.step()

            # Save the current uncertainty
            if self.epoch_tracking and epoch < n_epochs - 1:
                self.saved_models.append(copy.deepcopy(self.model))

            print(f"In epoch:{epoch} average loss was Loss: {avg_loss / n_batches}")

        # Reset device to cpu
        self.model.set_device(torch.device("cpu"))

        if dataset_name is not None:
            self.model.save(self.model.save_path + dataset_name + "/")

    def calibrate(
        self, X: np.ndarray | torch.Tensor, y: torch.Tensor | np.ndarray, alpha=0.1
    ):
        raise NotImplementedError

    def _predict(self, X: torch.Tensor, raw_outputs=False):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, X, raw_outputs=False):
        self.model.eval()
        if self.threshold is None:
            raise ValueError(
                "Model not calibrated. Please call the calibrate function before predicting."
            )

        if type(X) == DataLoader:
            output = [self._predict(inputs, raw_outputs=raw_outputs) for inputs, _ in X]
            return torch.cat(output)
        else:
            X = validate_predict_input(X, self.model.device)
            return self._predict(X, raw_outputs=raw_outputs)


class UACQR_S(UACQR):
    """
    Implements the UACQR method using Scores to calibrate the intervals.
    """

    def __init__(self, quantile_model: BaseDNN):
        super().__init__(quantile_model)
        self.threshold: float = None
        self.uncertainty_aware: bool = True

    def calibrate(
        self, X: np.ndarray | torch.Tensor, y: torch.Tensor | np.ndarray, alpha=0.1
    ):
        """
        Calibrate the non_conformity threshold
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        self.model.eval()

        if type(y) == torch.Tensor:
            y = to_numpy(y)

        intervals = np.zeros((len(self.saved_models), X.shape[0], 2))
        # gather the predictions from all models
        for i, model in enumerate(self.saved_models):
            model.eval()
            with torch.no_grad():
                predictions_test = model.predict(X)

            predictions_test = predictions_test.to(torch.device("cpu"))
            intervals[i] = to_numpy(predictions_test)

        non_conformity_scores = uacqr_s_score(intervals, y)

        # Get Quantile of non-conform scores

        self.threshold = calculate_quantile(non_conformity_scores, alpha=alpha)

    def _predict(self, X: torch.Tensor, raw_outputs=False):
        """
        Predict conformalized intervals, using the standard deviation of quantiles from the saved models.
        :param X: input data
        :param raw_outputs: If True, outputs the individual intervals of the models and the threshold, otherwise conformalized interval
        :return: conformalized interval or individual intervals of the models and the threshold, if raw_outputs is True
        """
        self.model.eval()

        intervals = np.zeros((len(self.saved_models), X.shape[0], 2))
        # gather the predictions from all models
        for i, model in enumerate(self.saved_models):
            model.eval()
            predictions_test = model.predict(X)

            predictions_test = predictions_test.to(torch.device("cpu"))
            intervals[i] = to_numpy(predictions_test)

        # calculate standard deviation of the predictions
        std = np.std(intervals, axis=0)
        predictions_test = np.mean(intervals, axis=0)
        g_low = std[:, 0]
        g_high = std[:, 1]
        cqr_interval = np.vstack(
            [
                predictions_test[:, 0] - self.threshold * g_low,
                predictions_test[:, 1] + self.threshold * g_high,
            ]
        ).T

        if raw_outputs:
            return intervals, self.threshold

        return cqr_interval


class UACQR_P(UACQR):
    """
    UACQR variant where the intervals are conformalized without score, but based on ensemble of saved quantile regressors.
    """

    def __init__(self, quantile_model: BaseDNN):
        super().__init__(quantile_model)

        self.uncertainty_aware: bool = True
        self.threshold: int = None

    def calibrate(
        self, X: np.ndarray | torch.Tensor, y: torch.Tensor | np.ndarray, alpha=0.1
    ):
        """
        Calibrate the non_conformity threshold
        :param X_cal: calibration data
        :param y_cal: calibration targets
        :param alpha: significance niveau, to construct sets with (marginal) cover of 1-\alpha
        :return: None
        """
        self.model.eval()

        if type(y) == torch.Tensor:
            y = to_numpy(y)

        intervals = np.zeros((len(self.saved_models), X.shape[0], 2))
        # gather the predictions from all models
        for i, model in enumerate(self.saved_models):
            model.eval()
            predictions_test = model.predict(X)

            predictions_test = predictions_test.to(torch.device("cpu"))
            intervals[i] = to_numpy(predictions_test)

        # Get quantiles for all models
        lower_values = intervals[:, :, 0]
        upper_values = intervals[:, :, 1]

        # Sort the models for each column according to the lower and upper values
        idx_lower = np.argsort(lower_values, axis=0)
        idx_upper = np.argsort(upper_values, axis=0)

        # Fill the resulting interval array
        res = np.zeros((len(self.saved_models) + 2, X.shape[0], 2))
        res[0] = np.vstack(
            [[np.inf], [-np.inf]]
        ).T  # Model 0 (smallest) --> Will not cover anything
        for i in range(1, len(self.saved_models) + 1):
            # We use different indexing as paper starts with 1 and coding starts with 0
            res[i] = np.vstack(
                [
                    # Get all values from the lower models
                    [
                        lower_values[
                            idx_lower[len(self.saved_models) - i],
                            np.arange(X.shape[0]),
                        ]
                    ],
                    # Get all values from the upper models
                    [upper_values[idx_upper[i - 1], np.arange(X.shape[0])]],
                ]
            ).T
        res[-1] = np.vstack(
            [[-np.inf], [np.inf]]
        ).T  # Model B+1 (biggest) --> Will cover everything

        # Get non-conform scores for all models
        y = y.flatten()
        non_conform_scores = np.maximum(
            res[:, :, 0] - y[None, :], y[None, :] - res[:, :, 1]
        )
        # non_conform_scores has shape (n_models, n_samples)
        contain_intervals = np.sum((non_conform_scores <= 0), axis=1) / (X.shape[0] + 1)
        # contain_intervals has shape (n_models,)

        # Filter the intervals that contain the quantile amount
        mask = contain_intervals >= (1 - alpha)
        # Get the uncertainty number that contains the quantile amount and set it to the uncertainty number
        self.threshold = np.argmax(mask)

    def _predict(self, X: torch.Tensor, raw_outputs=False):
        """
        Predict conformalized intervals, using the intervals of the saved models
        :param X: input data
        :param raw_outputs: If True, outputs the individual intervals of the models and the threshold, otherwise conformalized interval
        :return: conformalized interval or individual intervals of the models and the threshold, if raw_outputs is True
        """
        self.model.eval()
        predictions_test = self.model.predict(X)

        intervals = np.zeros((len(self.saved_models), X.shape[0], 2))
        # gather the predictions from all models
        for i, model in enumerate(self.saved_models):
            model.eval()
            predictions_test = model.predict(X)

            predictions_test = predictions_test.to(torch.device("cpu"))
            intervals[i] = to_numpy(predictions_test)

        lower_values = np.sort(intervals[:, :, 0], axis=0)
        upper_values = np.sort(intervals[:, :, 1], axis=0)

        # Get the correct uncertainty prediction
        if self.threshold == 0:
            prediction_interval = np.vstack(
                [np.full(X.shape[0], np.inf), np.full(X.shape[0], -np.inf)]
            ).T
        elif self.threshold == len(self.saved_models) + 1:
            prediction_interval = np.vstack(
                [np.full(X.shape[0], -np.inf), np.full(X.shape[0], np.inf)]
            ).T
        else:
            prediction_interval = np.vstack(
                [
                    lower_values[len(self.saved_models) - (self.threshold - 1)],
                    upper_values[self.threshold - 1],
                ]
            ).T

        if raw_outputs:
            return intervals, self.threshold
        return prediction_interval

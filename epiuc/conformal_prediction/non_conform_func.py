from typing import Optional

import numpy as np

#########################################
# Classification Non-Conformity Functions#
#########################################
def lac_non_conformity_scores(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_classes = probabilities.shape[1]

    # creates boolean array where True indicates the conditional probability value we need for nonconformity score.
    indices = (np.arange(n_classes))[None, :] == y[:, None]
    lac_nonconformity_scores = 1 - probabilities[indices]
    return lac_nonconformity_scores


def aps_non_conformity_scores(
    probabilities: np.ndarray, y: Optional[np.ndarray] = None, randomized: bool = False
) -> np.ndarray:
    ordered, indices = (
        np.sort(probabilities, kind="stable", axis=-1)[:, ::-1],
        np.argsort(probabilities, kind="stable", axis=1)[:, ::-1],
    )
    cumsum = np.cumsum(ordered, axis=-1)

    if y is None:
        if randomized:
            U = np.random.uniform(low=0, high=1, size=probabilities.shape)
        else:
            U = np.zeros_like(probabilities)

        ordered_scores = cumsum - ordered * U
        sorted_indices = np.argsort(indices, kind="stable", axis=-1)
        scores = np.take_along_axis(ordered_scores, sorted_indices, axis=-1)
    else:
        if randomized:
            U = np.random.uniform(low=0, high=1, size=probabilities.shape[0])
        else:
            U = np.zeros_like(probabilities.shape[0])

        relevant_index = np.where(indices == y[:, None])
        scores = cumsum[relevant_index] - U * ordered[relevant_index]
        print("Scores", scores.shape)
    return scores


def raps_non_conformity_scores(
    probabilities: np.ndarray,
    y: np.ndarray = None,
    lam: float = 0.001,
    k_reg: float = 4,
    randomized: bool = False,
) -> np.ndarray:
    ordered, indices = (
        np.sort(probabilities, kind="stable", axis=-1)[:, ::-1],
        np.argsort(probabilities, kind="stable", axis=1)[:, ::-1],
    )
    cumsum = np.cumsum(ordered, axis=-1)

    if y is None:
        if randomized:
            U = np.random.uniform(low=0, high=1, size=probabilities.shape)
        else:
            U = np.zeros_like(probabilities)

        ordered_scores = (
            cumsum
            - U * ordered
            + lam * np.maximum((np.arange(1, probabilities.shape[-1] + 1) - k_reg), 0)
        )
        sorted_indices = np.argsort(indices, kind="stable", axis=-1)
        raps_scores = np.take_along_axis(ordered_scores, sorted_indices, axis=-1)
    else:
        if randomized:
            U = np.random.uniform(low=0, high=1, size=probabilities.shape[0])
        else:
            U = np.zeros_like(probabilities.shape[0])

        relevant_index = np.where(indices == y[:, None])
        raps_scores = (
            cumsum[relevant_index]
            - U * ordered[relevant_index]
            + lam * np.maximum((relevant_index[1] + 1 - k_reg), 0)
        )
    return raps_scores


#########################################
# Regression Non-Conformity Functions   #
#########################################
def absolute_residual_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute the absolute residual score
    :param y_pred: predicted values
    :param y_true: true values
    :return: absolute residual score
    """
    return np.abs(y_pred - y_true)


def cqr_score(intervals: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute the CQR non-conformity-score
    :param intervals: quantile intervals of shape (n_samples, 2)
    :param y_true: true target values of shape (n_samples,) or (n_samples,1)
    :return: non-conformity score of shape (n_samples,)
    """
    if len(y_true.shape) == 2:
        y_true = y_true.flatten()

    lower_quantile = intervals[:, 0]
    upper_quantile = intervals[:, 1]

    cqr_score = np.maximum(lower_quantile - y_true, y_true - upper_quantile)

    return cqr_score


def cqr_r_score(intervals: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute the CQR-r non-conformity-score
    :param intervals: quantile intervals of shape (n_samples, 2)
    :param y_true: true target values of shape (n_samples,) or (n_samples,1)
    :return:  non-conformity score of shape (n_samples,)
    """
    if len(y_true.shape) == 2:
        y_true = y_true.flatten()

    width = intervals[:, 1] - intervals[:, 0]
    cqr_r_scores = np.maximum(
        (intervals[:, 0] - y_true) / width,
        (y_true - intervals[:, 1]) / width,
    )
    return cqr_r_scores


def uacqr_s_score(intervals: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute the UACQR_S non-conformity-score.
    :param intervals: intervals of shape (n_estimations, n_samples,2)
        The first dimension is then constructed e.g. by an ensemble of models
    :param y_true: true target values of shape (n_samples,) or (n_samples,1)
    :return: non-conformity score of shape (n_samples,)
    """
    if y_true.ndim == 2:
        y_true = y_true.flatten()
    # calculate standard deviation of the predictions
    std = np.std(intervals, axis=0)
    mean_intervals = np.mean(intervals, axis=0)

    quantile_std_low = std[:, 0]
    quantile_std_high = std[:, 1]

    lower_quantile = mean_intervals[:, 0]
    upper_quantile = mean_intervals[:, 1]

    non_conform_scores = np.maximum(
        (lower_quantile - y_true) / quantile_std_low,
        (y_true - upper_quantile) / quantile_std_high,
    )
    return non_conform_scores

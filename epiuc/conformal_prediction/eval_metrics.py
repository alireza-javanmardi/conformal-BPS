from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

########################################################
######Conformal Prediction Evaluation Metrics ##########
########################################################
def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    assert (
        y_true.shape == y_pred.shape
    ), "The shapes of the true and predicted values must be the same."
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _coverage_regression(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray):
    assert (
        y_lower.shape == y_upper.shape
    ), "The shapes of the lower and upper bounds must be the same."

    if len(y_true.shape) == 2:
        y_true = y_true.flatten()

    return np.sum((y_true >= y_lower) & (y_true <= y_upper)) / y_true.shape[0]


def _coverage_classification(y_true: np.ndarray, pred_interval: np.ndarray):
    if len(y_true.shape) == 2:
        y_true = y_true.flatten()

    return np.sum(pred_interval[np.arange(y_true.shape[0]), y_true]) / y_true.shape[0]


def coverage(y_true: np.ndarray, prediction_interval: np.ndarray, task: str):
    if type(prediction_interval) == torch.Tensor:
        prediction_interval = prediction_interval.cpu().numpy()

    if task == "regression":
        return _coverage_regression(
            y_true, prediction_interval[:, 0], prediction_interval[:, 1]
        )
    elif task == "classification":
        return _coverage_classification(y_true, prediction_interval)
    else:
        raise ValueError("The task must be either 'regression' or 'classification'.")


def average_set_size(y_upper: np.ndarray, y_lower: np.ndarray) -> float:
    if type(y_upper) == torch.Tensor:
        y_upper = y_upper.cpu().numpy()
    if type(y_lower) == torch.Tensor:
        y_lower = y_lower.cpu().numpy()

    return np.mean(abs(y_upper - y_lower))


def interval_score_loss(high_est, low_est, actual, alpha):
    return (
        high_est
        - low_est
        + 2 / alpha * (low_est - actual) * (actual < low_est)
        + 2 / alpha * (actual - high_est) * (actual > high_est)
    )


def average_interval_score_loss(high_est, low_est, actual, alpha):
    if type(high_est) == torch.Tensor:
        high_est = high_est.cpu().numpy()
    if type(low_est) == torch.Tensor:
        low_est = low_est.cpu().numpy()
    if type(actual) == torch.Tensor:
        actual = actual.cpu().numpy()

    return np.mean(interval_score_loss(high_est, low_est, actual, alpha))


# Evaluation Metrics TorchCP sligthly changed  #
# Source: https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/classification/utils/metrics.py
def covGap(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    num_classes: int,
    shot_idx: Optional[list] = None,
):
    """
    The average class-conditional coverage gap.

    Paper: Class-Conditional Conformal Prediction with Many Classes (Ding et al., 2023)
    Link: https://neurips.cc/virtual/2023/poster/70548

    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        labels (torch.Tensor): Ground-truth labels (N,).
        alpha (float): User-guided confidence level.
        num_classes (int): Number of classes.
        shot_idx (list, optional): Indices of classes to compute coverage gap.

    Returns:
        float: Average class-conditional coverage gap (percentage).
    """

    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")

    covered = prediction_sets[np.arange(len(labels)), labels]
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_covered = np.bincount(labels[covered], minlength=num_classes).astype(float)

    cls_coverage_rate = np.zeros_like(class_counts, dtype=np.float32)
    valid_classes = class_counts > 0
    cls_coverage_rate[valid_classes] = (
        class_covered[valid_classes] / class_counts[valid_classes]
    )

    if shot_idx is not None:
        cls_coverage_rate = cls_coverage_rate[shot_idx]

    overall_covgap = np.mean(np.abs(cls_coverage_rate - (1 - alpha))) * 100
    return overall_covgap

def covGap_nonbinary(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    num_classes: int,
    shot_idx: Optional[list] = None,
):
    """
    The average class-conditional coverage gap.

    Paper: Class-Conditional Conformal Prediction with Many Classes (Ding et al., 2023)
    Link: https://neurips.cc/virtual/2023/poster/70548

    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        labels (torch.Tensor): Ground-truth labels (N,).
        alpha (float): User-guided confidence level.
        num_classes (int): Number of classes.
        shot_idx (list, optional): Indices of classes to compute coverage gap.

    Returns:
        float: Average class-conditional coverage gap (percentage).
    """

    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")

    covered = prediction_sets[np.arange(len(labels)), labels]
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_covered = np.bincount(labels, weights= covered, minlength=num_classes).astype(float)

    cls_coverage_rate = np.zeros_like(class_counts, dtype=np.float32)
    valid_classes = class_counts > 0
    cls_coverage_rate[valid_classes] = (
        class_covered[valid_classes] / class_counts[valid_classes]
    )

    if shot_idx is not None:
        cls_coverage_rate = cls_coverage_rate[shot_idx]

    overall_covgap = np.mean(np.abs(cls_coverage_rate - (1 - alpha))) 
    return overall_covgap


def vio_classes(
    prediction_sets: np.ndarray, labels: np.ndarray, alpha: float, num_classes: int
):
    """
    Calculates the number of violated classes.
    :param prediction_sets: the one-hot encoded prediction sets
    :param labels: the true class
    :param alpha: the alpha value
    :param num_classes: the amount of classes
    :return:
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    violation_nums = 0
    for k in range(num_classes):
        if len(labels[labels == k]) == 0:
            violation_nums += 1
        else:
            idx = np.where(labels == k)[0]
            selected_preds = [prediction_sets[i] for i in idx]
            if (
                coverage(
                    labels[labels == k], np.array(selected_preds), "classification"
                )
                < 1 - alpha
            ):
                violation_nums += 1
    return violation_nums


def diff_violation(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    alpha: float,
    num_classes: int,
):
    """
    Calculates the difference violation for classification tasks.
    :param prediction_sets: the one-hot encoded prediction sets
    :param labels: the true classes
    :param logits: the logits
    :param alpha: the alpha value
    :param num_classes: the number of classes
    :return:
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    if labels.ndim == 2:
        labels = labels.flatten()

    strata_diff = [[1, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]
    correct_array = np.zeros(len(labels))
    size_array = np.zeros(len(labels))
    topk = []
    for index, ele in enumerate(logits):
        indices = np.argsort(ele)[::-1]
        target = labels[index]
        topk.append(np.where((indices - target.reshape(-1, 1)) == 0)[1] + 1)
        correct_array[index] = 1 if prediction_sets[index, labels[index]] else 0
        size_array[index] = np.sum(prediction_sets[index])
    topk = np.concatenate(topk)

    ccss_diff = {}
    diff_violation = -1

    for stratum in strata_diff:

        temp_index = np.argwhere((topk >= stratum[0]) & (topk <= stratum[1]))
        ccss_diff[str(stratum)] = {}
        ccss_diff[str(stratum)]["cnt"] = len(temp_index)
        if len(temp_index) == 0:
            ccss_diff[str(stratum)]["cvg"] = 0
            ccss_diff[str(stratum)]["sz"] = 0
        else:
            temp_index = temp_index[:, 0]
            cvg = np.round(np.mean(correct_array[temp_index]), 3)
            sz = np.round(np.mean(size_array[temp_index]), 3)

            ccss_diff[str(stratum)]["cvg"] = cvg
            ccss_diff[str(stratum)]["sz"] = sz
            stratum_violation = max(0, (1 - alpha) - cvg)
            diff_violation = max(diff_violation, stratum_violation)

    diff_violation_one = 0
    for i in range(1, num_classes + 1):
        temp_index = np.argwhere(topk == i)
        if len(temp_index) > 0:
            temp_index = temp_index[:, 0]
            stratum_violation = max(0, (1 - alpha) - np.mean(correct_array[temp_index]))
            diff_violation_one = max(diff_violation_one, stratum_violation)
    return diff_violation, diff_violation_one, ccss_diff


def size_stratified_cov_violation(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]],
):
    """
    Size-stratified coverage violation (SSCV)
    :param prediction_sets: the one-hot encoded prediction sets
    :param labels: the true classes
    :param alpha:  the alpha value
    :param stratified_size: list of size strata
    :return:
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = np.sum(ele)
        correct_array[index] = 1 if ele[labels[index]] else 0

    sscv = -1
    for stratum in stratified_size:
        temp_index = np.argwhere(
            (size_array >= stratum[0]) & (size_array <= stratum[1])
        )
        if len(temp_index) > 0:
            stratum_violation = abs((1 - alpha) - np.mean(correct_array[temp_index]))
            sscv = max(sscv, stratum_violation)
    return sscv

def size_stratified_cov_violation_nonbinary(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]],
):
    """
    Size-stratified coverage violation (SSCV)
    :param prediction_sets: the one-hot encoded prediction sets
    :param labels: the true classes
    :param alpha:  the alpha value
    :param stratified_size: list of size strata
    :return:
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    size_array = np.zeros(len(labels))
    coverage_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = np.sum(ele)
        coverage_array[index] = ele[labels[index]] 

    # sscv = -1
    # for stratum in stratified_size:
    #     temp_index = np.argwhere(
    #         (size_array >= stratum[0]) & (size_array <= stratum[1])
    #     )
    #     if len(temp_index) > 5:
    #         stratum_violation = abs((1 - alpha) - np.mean(coverage_array[temp_index]))
    #         sscv = max(sscv, stratum_violation)
    sscv = 2025
    for stratum in stratified_size:
        temp_index = np.argwhere(
            (size_array >= stratum[0]) & (size_array <= stratum[1])
        )
        if len(temp_index) > 10:
            stratum_violation = np.mean(coverage_array[temp_index])
            sscv = min(sscv, stratum_violation)
    return sscv

def uncertainty_stratified_cov_violation_nonbinary(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float,
    stratified_size=[[i, i+0.1] for i in np.arange(0,1,0.1)],
):
    """
    unertainty-stratified coverage violation (uscv)
    :param prediction_sets: the one-hot encoded prediction sets
    :param labels: the true classes
    :param uncertainty: the uncertainteis (could be total, epistemic, or aleatoic)
    :param alpha:  the alpha value
    :param stratified_size: list of size strata
    :return:
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    u_array = np.zeros(len(labels))
    coverage_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        u_array[index] = uncertainty[index]
        coverage_array[index] = ele[labels[index]] 

    # uscv = -1
    # for stratum in stratified_size:
    #     temp_index = np.argwhere(
    #         (u_array >= stratum[0]) & (u_array <= stratum[1])
    #     )
    #     if len(temp_index) > 5:
    #         stratum_violation = abs((1 - alpha) - np.mean(coverage_array[temp_index]))
    #         uscv = max(uscv, stratum_violation)
    uscv = 2025
    for stratum in stratified_size:
        temp_index = np.argwhere(
            (u_array >= stratum[0]) & (u_array <= stratum[1])
        )
        if len(temp_index) > 10:
            stratum_violation = np.mean(coverage_array[temp_index])
            uscv = min(uscv, stratum_violation)
    return uscv

def WSC(
    features,
    prediction_sets,
    labels,
    delta=0.1,
    M=1000,
    test_fraction=0.75,
    random_state=42,
    verbose=False,
):
    """
    Worst-Slice Coverage (WSC).

     Classification with Valid and Adaptive Coverage (Romano et al., 2020)
     Paper: Classification with Valid and Adaptive Coverage
     Link : https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
     Code: https://github.com/msesia/arc/tree/d80d27519f18b11e7feaf8cf0da8827151af9ce3


    Args:
        features (np.ndarray): Input features (N x D).
        prediction_sets (np.ndarray): Boolean tensor of prediction sets (N x C).
        y (np.ndarray): Ground-truth labels (N,).
        delta (float): Confidence level (between 0 and 1).
        M (int): Number of random projections.
        test_size (float): Proportion of tests split.
        random_state (int): Random seed.
        verbose (bool): Whether to print progress.

     Returns:
         Float: the value of unbiased WSV.

    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    if not 0 < test_fraction < 1:
        raise ValueError("test_size must be between 0 and 1")
    if M <= 0:
        raise ValueError("M must be positive")

    if len(features.shape) != 2:
        raise ValueError(f"features must be 2D tensor, got shape {features.shape}")
    if len(prediction_sets.shape) != 2:
        raise ValueError(
            f"prediction_sets must be 2D tensor, got shape {prediction_sets.shape}"
        )
    if len(labels.shape) != 1:
        raise ValueError(f"labels must be 1D tensor, got shape {labels.shape}")

    if features.shape[0] != len(labels):
        raise ValueError(
            f"Number of samples mismatch: features has {features.shape[0]} samples but labels has {len(labels)} samples"
        )
    if features.shape[0] != prediction_sets.shape[0]:
        raise ValueError(
            f"Number of samples mismatch: features has {features.shape[0]} samples but prediction_sets has {prediction_sets.shape[0]} samples"
        )
    if prediction_sets.shape[1] != len(np.unique(labels)):
        raise ValueError(
            f"Number of classes mismatch: prediction_sets has {prediction_sets.shape[1]} classes but labels has {len(np.unique(labels))} unique classes"
        )

    covered = prediction_sets[np.arange(len(labels)), labels]

    X_train, X_test, y_train, y_test, covered_train, covered_test = train_test_split(
        features, labels, covered, test_size=test_fraction, random_state=random_state
    )
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = _calWSC(
        X_train,
        covered_train,
        y_train,
        delta=delta,
        M=M,
        random_state=random_state,
    )
    # Estimate coverage
    coverage = _wsc_vab(X_test, y_test, covered_test, v_star, a_star, b_star)
    return coverage


def _wsc_vab(featreus, labels, covered, v, a, b):
    z = np.dot(featreus, v)
    idx = (z >= a) & (z <= b)
    return np.mean(covered[idx])


def _calWSC(X, y, covered, delta=0.1, M=1000, random_state=2020):
    rng = np.random.default_rng(random_state)
    n = len(y)

    def wsc_v(X, covered, delta, v):
        z = np.dot(X, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = covered[z_order]

        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n - 1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n - 1)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n: int, p: int) -> np.ndarray:
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    results = np.zeros((M, 4))

    iterator = range(M)

    for m in iterator:
        wsc, a, b = wsc_v(X, covered, delta, V[m])
        results[m] = [wsc, a, b, m]

    idx_best = np.argmin(results[:, 0])
    wsc_star = results[idx_best, 0]
    a_star = results[idx_best, 1]
    b_star = results[idx_best, 2]
    v_star = V[int(results[idx_best, 3])]

    return wsc_star, v_star, a_star, b_star


def singleton_hit_ratio(prediction_sets: np.ndarray, labels: np.ndarray):
    """
    Singleton hit ratio (SHR) for classification tasks.
    :param prediction_sets: one-hot encoded prediction sets
    :param labels: the true classes
    :return: SHR
    """
    if type(prediction_sets) == torch.Tensor:
        prediction_sets = prediction_sets.cpu().numpy()
    if len(prediction_sets) == 0:
        raise AssertionError("The number of prediction set must be greater than 0.")
    n = len(prediction_sets)
    singletons = np.sum(prediction_sets, axis=1) == 1
    covered = prediction_sets[np.arange(len(labels)), labels]

    return np.sum(singletons & covered) / n

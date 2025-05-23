import matplotlib.pyplot as plt
import numpy as np
import torch

from epiuc.utils.general import to_numpy


def plot_prediction_interval(ax, x, lower_bound, upper_bound, color, label, alpha):
    ax.scatter(x, upper_bound, s=1.0, c=color, zorder=0, label=label)
    ax.scatter(x, lower_bound, s=1.0, c=color, zorder=0)
    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        alpha=alpha,
        edgecolor=None,
        facecolor=color,
        linewidth=0,
        zorder=1,
    )


def plot_prediction_intervals(
    x_train, y_train, x_test, intervals, colors, labels, alphas
):
    """
    Plot the prediction intervals
    :param x_train: Training feature data
    :param y_train: Training target data
    :param x_test: Test feature data
    :param intervals: List of prediction intervals
    :param colors: Colors for each interval
    :param labels: Names of the intervals
    :param alphas: Opacity of the intervals
    :return:
    """

    n_rows = len(intervals) // 3 + (1 if len(intervals) % 3 != 0 else 0)
    n_cols = 3 if len(intervals) > 3 else len(intervals)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2 * n_rows), dpi=200)

    idxs = x_test[:, 0].argsort()

    for i, interval in enumerate(intervals):

        if n_rows == 1 and n_cols == 1:
            ax.scatter(
                x_train[:, 0], y_train, s=1.0, c="#463c3c", zorder=0, label="Train"
            )
            ax.set_xlabel("$X_{\cdot,1}$")
            ax.set_ylabel("Y")
            ax.set_title("Y versus X")
            plot_prediction_interval(
                ax,
                x_test[:, 0][idxs],
                interval[:, 0][idxs],
                interval[:, 1][idxs],
                alpha=alphas[i],
                color=colors[i],
                label=labels[i],
            )
            ax.legend()
        elif n_rows == 1:
            ax[i % n_cols].scatter(
                x_train[:, 0], y_train, s=1.0, c="#463c3c", zorder=0, label="Train"
            )
            ax[i % n_cols].set_xlabel("$X_{\cdot,1}$")
            ax[i % n_cols].set_ylabel("Y")
            ax[i % n_cols].set_title("Y versus X")

            if labels[0] == "Test":
                plot_prediction_interval(
                    ax[i % n_cols],
                    x_test[:, 0][idxs],
                    intervals[0][:, 0][idxs],
                    intervals[0][:, 1][idxs],
                    alpha=alphas[0],
                    color=colors[0],
                    label=labels[0],
                )

            plot_prediction_interval(
                ax[i % n_cols],
                x_test[:, 0][idxs],
                interval[:, 0][idxs],
                interval[:, 1][idxs],
                alpha=alphas[i],
                color=colors[i],
                label=labels[i],
            )
            ax[i % n_cols].legend()
        else:
            ax[i // n_cols, i % n_cols].scatter(
                x_train[:, 0], y_train, s=1.0, c="#463c3c", zorder=0, label="Train"
            )
            ax[i // n_cols, i % n_cols].set_xlabel("$X_{\cdot,1}$")
            ax[i // n_cols, i % n_cols].set_ylabel("Y")
            ax[i // n_cols, i % n_cols].set_title("Y versus X")

            if labels[0] == "Test":
                plot_prediction_interval(
                    ax[i // n_cols, i % n_cols],
                    x_test[:, 0][idxs],
                    intervals[0][:, 0][idxs],
                    intervals[0][:, 1][idxs],
                    alpha=alphas[0],
                    color=colors[0],
                    label=labels[0],
                )

            plot_prediction_interval(
                ax[i // n_cols, i % n_cols],
                x_test[:, 0][idxs],
                interval[:, 0][idxs],
                interval[:, 1][idxs],
                alpha=alphas[i],
                color=colors[i],
                label=labels[i],
            )
            ax[i // n_cols, i % n_cols].legend()
        fig.tight_layout(h_pad=2, w_pad=1)


def plot_interval(
    x_train,
    y_train,
    x_test,
    y_test,
    interval,
    x_min_train,
    x_max_train,
    x_min_test,
    x_max_test,
    y_min,
    y_max,
):
    x_test = x_test[:, 0]
    if type(interval) == torch.Tensor:
        lower_bound, upper_bound = torch.split(interval, 1, dim=-1)
        # make every tensor be one-dimensional
        lower_bound = lower_bound.squeeze()
        upper_bound = upper_bound.squeeze()
        y_pred = interval.mean(dim=-1)
    else:
        lower_bound, upper_bound = interval[:, 0], interval[:, 1]
        y_pred = (upper_bound + lower_bound) / 2

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1.0, c="#463c3c", zorder=0, label="Train")
    plt.plot(x_test, y_test, "r--", zorder=2, label="True")
    plt.plot(x_test, y_pred, color="#007cab", zorder=3, label="Pred")
    plt.plot([x_min_train, x_min_train], [y_min, y_max], "k--", alpha=0.4, zorder=0)
    plt.plot([x_max_train, x_max_train], [y_min, y_max], "k--", alpha=0.4, zorder=0)
    plt.fill_between(
        x_test,
        lower_bound,
        upper_bound,
        alpha=0.3,
        edgecolor=None,
        facecolor="#00aeef",
        linewidth=0,
        zorder=1,
        label="Unc.",
    )
    plt.gca().set_ylim(y_min, y_max)
    plt.gca().set_xlim(x_min_test, x_max_test)
    plt.legend(loc="upper left")
    plt.show()


def plot_uncertainty_predictions(
    x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0
):

    x_test = x_test[:, 0]
    mu, var = y_pred
    mu = to_numpy(mu)
    var = to_numpy(var)
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    mu = mu[:, 0]

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1.0, c="#463c3c", zorder=0, label="Train")
    plt.plot(x_test, y_test, "r--", zorder=2, label="True")
    plt.plot(x_test, mu, color="#007cab", zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], "k--", alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], "k--", alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test,
            (mu - k * var),
            (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor="#00aeef",
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None,
        )
    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()


def add_plot_eval_rotation_proba(axes, rotated_proba, degrees):
    """
    Add a rotation classification plot to the given axes
    :param axes:
    :param rotated_proba:
    :param degrees:
    :return:
    """
    n_classes = rotated_proba.shape[-1]

    axes.set_title("Classification Probability over Rotation Degrees")

    for c in range(n_classes):
        axes.plot(degrees, rotated_proba[:, c], marker="x", label=f"Proba. Class {c}")

    axes.set_xlabel("Rotation Degrees")
    axes.set_ylabel("Classification Probabilities")

    axes.legend()


def add_plot_eval_rotation_uncertainty(axes, uncertainty, degrees):
    """
    Add a rotation uncertainty plot to the given axes
    :param axes: axes to plot on
    :param uncertainty: the uncertainty values for each degree
    :param degrees: the degrees that were evaluated
    :return: Nothing
    """
    axes.set_title("Uncertainty over Rotation Degrees")

    axes.plot(degrees, uncertainty, marker="s")

    axes.set_xlabel("Rotation Degrees")
    axes.set_ylabel("Classification Probabilities")


def plot_eval_rotation(rotated_proba, degrees, uncertainties, model_names):
    """
    Plot the evaluation results of the rotation classification
    :param rotated_proba: probability of classes for each rotation degree
    :param degrees: degrees that were evaluated
    :param uncertainties: uncertainty of the models for each degree
    :param model_names: name of the models
    :return: Nothing
    """

    n_models, _, _ = rotated_proba.shape
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    fig.suptitle("Results for One-Image with ...", fontweight="bold")
    subfigs = fig.subfigures(nrows=n_models, ncols=1)

    for i in range(n_models):
        subfigs[i].suptitle(f"... {model_names[i]}", fontweight="bold")
        if not np.all(np.isnan(uncertainties[i])):
            axes = subfigs[i].subplots(nrows=1, ncols=2)
            add_plot_eval_rotation_proba(axes[0], rotated_proba[i], degrees[i])
            add_plot_eval_rotation_uncertainty(axes[1], uncertainties[i], degrees[i])
        else:
            axes = subfigs[i].subplots(nrows=1, ncols=1)
            add_plot_eval_rotation_proba(axes, rotated_proba[i], degrees[i])

    plt.show()

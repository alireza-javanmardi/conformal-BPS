import numpy as np
import torch

from epiuc.utils.general import to_numpy, rotate_img, gather_results
from epiuc.utils.plotting import plot_eval_rotation


######################################################
############# General Evaluation Metrics #############
######################################################


def evaluate_auroc(models, testloader):
    """
    Evaluate the Binary AUROC of given models on testloader
    :param models:
    :param testloader:
    :return:
    """
    n_samples = len(testloader.sampler)
    y_true = np.zeros((len(models), n_samples))
    y_score = np.zeros((len(models), n_samples))

    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            for j, model in enumerate(models):
                prediction = model.predict(input)
                if model.uncertainty_aware:
                    probs, _ = prediction
                    y_score[j, i] = to_numpy(probs)[:, 1]
                else:
                    probs = prediction
                    y_score[j, i] = to_numpy(probs)[:, 1]

                y_true[j, i] = to_numpy(target)

    # compute binary auroc
    from sklearn.metrics import roc_auc_score

    rocs_aucs = [roc_auc_score(y_true[i], y_score[i]) for i in range(len(models))]
    return rocs_aucs


def evaluate_accuracy(models, testloader):
    n_samples = len(testloader.sampler)

    accuracies = []

    with torch.no_grad():
        for i, model in enumerate(models):
            model.eval()
            probabilties, _, _, target = gather_results(model, testloader)

            y_pred = np.argmax(probabilties, axis=-1)
            correct = np.sum(y_pred == target)
            accuracies.append(correct / n_samples)

    return accuracies  #


def evaluate_rotation_proba(start_image, models):
    """
    Evaluate the classification probabilities of a given image over a range of rotation degrees
    :param start_image: image to be rotated
    :param models: list of models
    :return: returns tuple consisting of rotated_pred_proba, evaluated_degrees, uncertainties
    """
    # Degree Predictions
    maximal_degree = 180
    n_classes = 10

    n_degrees = (maximal_degree // 10) + 1
    rotated_pred_proba = np.zeros((len(models), n_degrees, n_classes))
    evaluated_degrees = np.zeros((len(models), n_degrees))
    uncertainties = np.zeros((len(models), n_degrees))

    for i, degree in enumerate(
        np.linspace(start=0, stop=maximal_degree, num=n_degrees)
    ):

        rotated_one_image = rotate_img(
            start_image,
            deg=degree,
            output_width=start_image.shape[2],
            output_height=start_image.shape[3],
        )
        for j, model in enumerate(models):
            uncertainty = np.nan
            if model.uncertainty_aware:
                probability, uncertainty = model.predict(rotated_one_image)

                probability = to_numpy(probability).flatten()
                uncertainty = to_numpy(uncertainty).flatten()[0]

            else:
                probability = to_numpy(model.predict(rotated_one_image)).flatten()

            # Store results
            rotated_pred_proba[j, i] = probability
            evaluated_degrees[j, i] = degree
            uncertainties[j, i] = uncertainty

    plot_eval_rotation(
        rotated_pred_proba,
        evaluated_degrees,
        uncertainties,
        model_names=[m.name for m in models],
    )

    return rotated_pred_proba, evaluated_degrees, uncertainties

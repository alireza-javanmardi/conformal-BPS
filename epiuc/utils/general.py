from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import torch


@torch.no_grad()
@torch.inference_mode()
def gather_results(model, dataloader):
    """
    Function to gather probability,uncertainty, features and labels from a dataloader
    :param model:
    :param dataloader:
    :return: probabilities, uncertainty, features, y
    """
    uncertainty = None
    probabilities = None
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
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)

        if model.uncertainty_aware:
            prob, unc = model.predict(inputs)
            # move to cpu
            prob = prob.cpu().numpy()
            unc = unc.cpu().numpy()
        else:
            prob = model.predict(inputs)
            prob = prob.cpu().numpy()
            unc = None

        # Reset Padding
        inputs = inputs[:current_batch_size]
        if i == 0:
            probabilities, uncertainty = prob, unc

            probabilities = probabilities[:current_batch_size]

            if uncertainty is not None:
                uncertainty = uncertainty[:current_batch_size]

            features = (
                inputs.to(torch.device("cpu")).numpy().reshape(inputs.shape[0], -1)
            )
            y = labels.to(torch.device("cpu")).numpy()
        else:

            prob = prob[:current_batch_size]

            probabilities = np.vstack([probabilities, prob])
            if unc is not None:
                unc = unc[:current_batch_size]
                uncertainty = np.vstack([uncertainty, unc])
            features = np.vstack(
                [
                    features,
                    inputs.to(torch.device("cpu")).numpy().reshape(inputs.shape[0], -1),
                ]
            )
            y = np.hstack([y, labels.to(torch.device("cpu")).numpy()])
        i += 1
    if uncertainty is None:
        uncertainty = np.zeros((probabilities.shape[0], 1))

    return probabilities, uncertainty, features, y


def validate_predict_input(
    X: torch.Tensor | np.ndarray, device: Optional[torch.device] = torch.device("cpu")
) -> torch.Tensor:
    """
    Function to validate the input of the predict function
    :param X: input data
    :param device: device to move the data to if it is a numpy array
    :return: input data as torch.Tensor
    """
    if type(X) == np.ndarray:
        X = torch.from_numpy(X).float()
    elif type(X) != torch.Tensor:
        raise TypeError("Input should be a numpy array or a torch tensor")

    if X.device != device:
        X = X.to(device)

    return X


def validate_mlp_params(
    n_layers: int,
    n_neurons: int,
    dropout_rate: float,
    input_features: int,
    n_classes: int,
):
    """
    Function to validate the parameters of the MLP.

    Raises error if the parameters are not sensible/valid.
    :param n_layers: number of layers
    :param n_neurons: number of neurons
    :param dropout_rate: dropout rate
    :param input_features: number of input features
    :param n_classes: number of classes
    :return: None
    """
    if n_layers < 1:
        raise ValueError("n_layers must be greater than 0")
    if n_neurons < 1:
        raise ValueError("n_neurons must be greater than 0")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate must be between 0 and 1")
    if input_features < 1:
        raise ValueError("input_features must be greater than 0")
    if n_classes < 1:
        raise ValueError("n_classes must be greater than 0")


def validate_quantiles(quantiles: list[float]):
    """
    Function to validate the quantiles
    :param quantiles: list of quantiles the uncertainty shoud learn
    :return: None
    """
    if len(quantiles) < 1:
        raise ValueError("Quantiles must be a list of at least one value")
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("Quantiles must be unique values")
    if not all(isinstance(q, float) for q in quantiles):
        raise ValueError("Quantiles must be float values")


def construct_mlp(
    input_features: int,
    n_outputs: int,
    n_layers: int,
    n_neurons: int,
    dropout_rate: float = 0.5,
    batch_norm: bool = True,
) -> torch.nn.Sequential:
    """
    Function to construct an MLP uncertainty
    :param input_features: number of input features
    :param n_outputs: number of classes
    :param n_layers: number of layers
    :param n_neurons: number of neurons
    :param dropout_rate: dropout rate
    :param batch_norm: batch normalization
    :return: MLP uncertainty
    """
    validate_mlp_params(n_layers, n_neurons, dropout_rate, input_features, n_outputs)

    layers = []
    input_size = input_features
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(in_features=input_size, out_features=n_neurons))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(n_neurons))
        layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            layers.append(torch.nn.Dropout(p=dropout_rate))
        input_size = n_neurons
    layers.append(torch.nn.Linear(in_features=input_size, out_features=n_outputs))

    return torch.nn.Sequential(*layers)


def min_max_normalize(X: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Function to normalize the input data
    :param X: input data
    :return: normalized data and dictionary with min and max values
    """
    if type(X) == np.ndarray:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        # Avoid division by zero
        max_val = np.where(max_val == 0, np.ones_like(max_val), max_val)
        # Normalize
        X = (X - min_val) / (max_val - min_val)
    elif type(X) == torch.Tensor:
        min_val = torch.min(X, dim=0).values
        max_val = torch.max(X, dim=0).values
        # Avoid division by zero
        max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)
        # Normalize
        X = (X - min_val) / (max_val - min_val)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")
    normalisation_params = {
        "min": min_val,
        "max": max_val,
    }
    return X, normalisation_params


def pad_inputs(batch_size, inputs, labels):
    # Every data instance is an input + label pair
    current_batch_size, remain = inputs.shape[0], inputs.shape[1:]
    target_size = labels.shape[1:]
    # Pad the input to have same batch size
    inputs = torch.cat(
        [inputs, torch.zeros(batch_size - current_batch_size, *remain)],
        dim=0,
    )
    labels = torch.cat(
        [
            labels,
            torch.zeros(
                batch_size - current_batch_size,
                *target_size,
                dtype=torch.long,
            ),
        ],
        dim=0,
    )
    return inputs, labels


def set_model_to_eval(model):
    """
    Function to set the uncertainty to eval mode
    :param model: uncertainty to set to eval mode
    :return: None
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
    elif isinstance(model, list):
        for m in model:
            m.eval()


def get_in_features_last_linear_layer(model):
    # find the last linear layer
    model_children = list(model.children())[::-1]
    idx_remove_layers = 0
    for i, layer in enumerate(model_children):
        if isinstance(layer, torch.nn.Linear):
            idx_remove_layers = i
            break
    head_in_features = layer.in_features
    return head_in_features, idx_remove_layers, model_children


def set_seeds(seed: int) -> None:
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def calculate_quantile(
    non_conformity_scores: torch.Tensor | np.ndarray, alpha: float
) -> float:
    if type(non_conformity_scores) == torch.Tensor:
        non_conformity_scores = to_numpy(non_conformity_scores)
    quantile = np.quantile(non_conformity_scores, 1 - alpha, method="inverted_cdf")
    return quantile


def to_numpy(tensor):
    return tensor.numpy()


def extract_image(tensor_image):
    # return the image contained in the "first" batch and first channel
    return tensor_image[0][0]


def plot_image(image):
    plt.imshow(extract_image(image), cmap="gray", interpolation="none")
    plt.show()


def rotate_img(image, deg, output_width, output_height):
    result_shape = image.shape

    # Rotation counterclockwise
    rotated_image = (
        nd.rotate(extract_image(image), deg, reshape=False, cval=-1)
        .ravel()
        .reshape(output_width, output_height)
    )

    # Clip values between [-1, +1]
    rotated_image = np.clip(rotated_image, a_min=-1, a_max=1)

    # return Tensor
    return torch.Tensor(rotated_image).view(result_shape)


def seed_worker(worker_id):
    """
    Seed worker function for DataLoader to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


def find_device():
    """
    Find the best device to train on
    :return: torch.device
    """

    # Device Finding
    if torch.cuda.is_available():
        print("Using Cuda for training")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS Device for training")
        return torch.device("cpu")
    else:
        print("Using CPU for training")
        return torch.device("cpu")

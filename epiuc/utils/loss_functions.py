import math
import torch
import numpy as np


def validate_quantile_loss_params(output, target, quantiles):
    """
    Function to validate the parameters of the quantile loss function.

    Raises error if the parameters are not sensible/valid.
    :param output: The output of the uncertainty.
    :param target: The target tensor.
    :param quantiles: The quantiles the uncertainty should learn.
    :return: The target tensor with the same shape as the output tensor.
    """
    assert isinstance(output, torch.Tensor), "Predictions must be a torch.Tensor"
    assert isinstance(target, torch.Tensor), "Target must be a torch.Tensor"
    assert isinstance(
        quantiles, (list, torch.Tensor)
    ), "Quantiles must be a list or torch.Tensor"
    assert (
        len(output.shape) == 2
    ), "Predictions must have 2 dimensions (batch_size, num_quantiles)"
    assert output.shape[1] == len(
        quantiles
    ), f"Number of predictions ({output.shape[1]}) must match the number of quantiles ({len(quantiles)})"

    if type(target) is not torch.Tensor:
        if type(target) == np.ndarray:
            target = torch.from_numpy(target).float().to(output.device)
        else:
            raise ValueError(
                "Target must be a torch.Tensor or a numpy.ndarray. "
                f"Got {type(target)} instead."
            )
    if output.shape[-1] != target.shape[-1]:
        # Expand the target to match the output shape
        if type(target) == torch.Tensor:
            target = target.expand(-1, output.shape[-1])

    # Convert quantiles to a tensor if it's a list
    if isinstance(quantiles, list):
        quantiles_tensor = torch.tensor(quantiles, device=output.device).view(1, -1)
    else:
        quantiles_tensor = quantiles.view(1, -1)
    return output, target, quantiles_tensor


def quantile_loss(
    output: torch.Tensor, target: torch.Tensor, quantiles: list
) -> torch.Tensor:

    output, target, quantiles_tensor = validate_quantile_loss_params(
        output, target, quantiles
    )

    # Calculate errors
    errors = target - output

    # Calculate losses for each quantile
    losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)

    # Sum the losses and take the mean
    loss = torch.mean(torch.sum(losses, dim=1))

    return loss


def sigma_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:

    logprob = (
        -(sigma).log()
        - 0.5 * math.log(2 * torch.pi)
        - 0.5 * ((target - mu) / sigma) ** 2
    )

    return -logprob.mean()

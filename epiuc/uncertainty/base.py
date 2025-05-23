"""
Contains the base class for all deep learning models.
"""
import functools
import os
import warnings
from copy import deepcopy
from typing import Optional, Any, Union

import numpy as np
import torch
from torch import vmap, Tensor
from torch._functorch.functional_call import stack_module_state, functional_call
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from epiuc.CustomDataset import CustomDataset
from epiuc.utils.general import (
    set_seeds,
    find_device,
    pad_inputs,
    validate_predict_input,
)
from epiuc.config import (
    BACKEND,
    DATALOADER_CONFIGS,
    DEFAULT_CONFIGS,
    OPTIMIZER_CONFIGS,
    SCHEDULER_CONFIGS,
)


def _convert_to_dataloader(
    X: np.ndarray | torch.Tensor | torch.utils.data.DataLoader,
    Y: Optional[np.ndarray],
    batch_size: int = DATALOADER_CONFIGS["batch_size"],
    shuffle: bool = DATALOADER_CONFIGS["shuffle"],
) -> Optional[torch.utils.data.DataLoader]:
    """
    Function to convert training params to a Dataloader object
    :param  X: features or complete data loader
    :param  Y: labels or None
    :param  batch_size: batch size for the dataloader
    :param shuffle: whether to shuffle the data
    :return: A data loader object or None if no data is provided
    """
    dataloader = None

    if X is None and Y is None:
        return dataloader
    if Y is None:
        if not isinstance(X, torch.utils.data.DataLoader):
            # If only X is provided, we construct a dataloader with X as input and as output we only have -1.
            # We inherently that the output will be never used
            dataset = CustomDataset(X, -1 * np.ones(X.shape[0]))
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=DATALOADER_CONFIGS["num_workers"],
                pin_memory=DATALOADER_CONFIGS["pin_memory"],
            )
        else:
            dataloader = X
    else:
        if isinstance(X, torch.utils.data.DataLoader):
            raise ValueError(
                "If Y is not None, X must be a numpy array or torch tensor."
            )
        dataset = CustomDataset(X, Y)
        dataloader = DataLoader(
            dataset,
            batch_size=DATALOADER_CONFIGS["batch_size"],
            shuffle=DATALOADER_CONFIGS["shuffle"],
            num_workers=DATALOADER_CONFIGS["num_workers"],
            pin_memory=DATALOADER_CONFIGS["pin_memory"],
        )
    return dataloader


class BaseDNN(torch.nn.Module):
    """
    Base class for all deep learning models.
    This class contains the core training process and the predict function.
    It also contains the function to save and load the uncertainty weights.

    """

    def __init__(self, model: torch.nn.Module, random_state: Optional[int] = None):
        """
        Constructor for the BaseDNN class.
        :param model: The uncertainty to be used. This should be a torch.nn.Module object.
        :param random_state: Random state for reproducibility. Default is None.
        """
        if random_state:
            set_seeds(random_state)
        super(BaseDNN, self).__init__()

        self.model = deepcopy(model)

        # Will be set when called in fit
        self.current_epoch: Optional[int] = None
        self.n_batches: int = 0

        # General Information
        self.name: str = "BaseDNN"
        self.uncertainty_aware: bool = False
        self.sigma: bool = False
        self.quantile: bool = False
        self.quantiles: Optional[list[float]] = None

        # Training Information
        self.save_path: str = "models/"
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler = None

        # Device Finding
        self.device: torch.device = torch.device("cpu")

        # Saving the best uncertainty weights in validation
        self.best_model_weights = None

    def set_device(self, device) -> None:
        """
        Function to set the device of the uncertainty
        :param  device: device to set the uncertainty to
        :return: None
        """
        self.to(device)
        self.device = device

    def loss_function(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Function to compute the loss for the uncertainty to be optimized during training.
        :param  output: the models output for a given input
        :param  target: the true values of the corresponding output
        :param  kwargs: parameters to be passed to the loss function
        :return: loss
        """
        raise NotImplementedError(
            "This function needs to be overwritten in the child classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to compute the forward pass of the uncertainty
        :param x: input tensor
        :return: prediction tensor
        """
        x = self.model(x)
        return x

    def train_step(
        self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Function to perform a single training step for the uncertainty given the inputs and labels
        :param inputs: The input data
        :param labels: The true labels
        :param kwargs: Additional parameters to pass to the training step
        :return: The loss value
        """
        # Zero your gradients for every batch!
        self.optimizer.zero_grad(set_to_none=True)

        # Compute forward pass
        outputs = self(inputs)

        # Computes loss and gradient
        loss = self.loss_function(outputs, labels, **kwargs)
        if loss.isnan():
            print(
                f"Loss is NaN at epoch:{kwargs['epoch']} and iter:{kwargs['iterpoch']}"
            )
            raise ValueError("Loss is NaN")
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()
        return loss

    def val_step(
        self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Function to perform a single validation step for the uncertainty given the inputs and labels.
        Deactivates the dropout layers and batch normalization layers.
        :param inputs: The input data
        :param labels: The true labels
        :param kwargs: Additional parameters to pass to the validation step
        :return: The loss value
        """
        with torch.no_grad():
            # Deactivate dropout layers and batch normalization layers
            self.eval()
            outputs = self(inputs)

            # Computes loss
            loss = self.loss_function(outputs, labels, **kwargs)

        return loss

    def update_best_model(
        self, valloader: torch.utils.data.DataLoader, best_val_loss: float, epoch: int
    ) -> float:
        """
        Function to update the best uncertainty weights based on the validation loss

        :param valloader: dataloader containing the validation data
        :param best_val_loss: best validation loss so far
        :param epoch: current epoch number
        :param n_batches: amount of batches in the validation data
        :return: Best validation loss so far
        """
        # This self.eval() might be changed for MC Dropout.
        self.eval()
        n_batches = len(valloader)
        val_loss = 0

        for i, data in enumerate(valloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            val_loss += self.val_step(
                inputs,
                labels,
                **{"epoch": epoch, "iterpoch": i, "n_batches": n_batches},
            )
        print(
            f"In epoch:{epoch} average validation loss was Loss: {val_loss / n_batches}"
        )

        if val_loss < best_val_loss:
            self.best_model_weights = deepcopy(self.state_dict())
            return val_loss
        return best_val_loss

    def _training_process(
        self,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader = None,
        n_epochs: int = 50,
    ) -> None:
        """
        Function performing the core training process of the uncertainty.
        :param trainloader: training data loader
        :param valloader: validation data loader
        :param n_epochs: number of epochs
        :return: Saves the uncertainty to disk and returns None
        """
        # Setup
        n_batches = len(trainloader)
        self.set_device(find_device())
        self.compile(backend=BACKEND)

        # Tmp variables
        best_val_loss = float("inf")
        device = self.device
        batch_size = trainloader.batch_size

        for epoch in range(n_epochs):

            self.train(True)

            avg_loss = 0

            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = pad_inputs(batch_size, inputs, labels)

                # Move inputs und labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                avg_loss += self.train_step(
                    inputs,
                    labels,
                    **{"epoch": epoch, "iterpoch": i, "n_batches": n_batches},
                )

                # print(f"Finished Batch {i} from {n_batches}")

            # Adjust learning rate
            self.scheduler.step()

            print(f"In epoch:{epoch} average loss was Loss: {avg_loss / n_batches}")

            if valloader is not None:
                best_val_loss = self.update_best_model(valloader, best_val_loss, epoch)

        self.device = torch.device("cpu")
        self.set_device(self.device)

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor | torch.utils.data.DataLoader,
        Y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray | torch.Tensor | torch.utils.data.DataLoader] = None,
        Y_val: Optional[np.ndarray] = None,
        dataset_name: Optional[str] = None,
        n_epochs: int = DEFAULT_CONFIGS["n_epochs"],
        batch_size: int = DATALOADER_CONFIGS["batch_size"],
        optimizer_params: dict[str, int] = OPTIMIZER_CONFIGS["adam"],
        scheduler_params: dict[str, int] = SCHEDULER_CONFIGS["step"],
    ):
        """
        Function to fit the deep learning uncertainty
        This function will also save the uncertainty to disk if a dataset name is provided.
        :param  X_train: features or complete data loader
        :param  Y_train: labels or None
        :param  X_val: features or complete data loader
        :param  Y_val: labels or None
        :param  dataset_name: name of the dataset to save the uncertainty. If None the uncertainty will not be saved to an folder, otherwise it will be saved to the path models/dataset_name/
        :param  n_epochs: number of epochs
        :param  batch_size: batch size for the dataloader to be used, if torch.Tensor or numpy array is given
        :param  optimizer_params: dictionary containing the optimizer and its parameters in the field "optimizer" and "params" respectively. See `config.py` for more information.
        :param  scheduler_params: dictionary containing the scheduler and its parameters in the field "scheduler" and "params" respectively. See `config.py` for more information.
        :return: self

        """

        self.init_optimizer_scheduler(optimizer_params, scheduler_params)

        trainloader = _convert_to_dataloader(X_train, Y_train, batch_size)
        valloader = _convert_to_dataloader(X_val, Y_val, batch_size)

        self._training_process(trainloader, valloader, n_epochs)

        # Reset current epoch
        self.current_epoch = None

        # Load the best uncertainty weights if they exist
        if self.best_model_weights is not None:
            self.load_state_dict(self.best_model_weights)

        # Save the uncertainty to disk if a dataset name is provided
        if dataset_name is not None:
            self.save(self.save_path + dataset_name + "/")

        return self

    @torch.no_grad()
    def _predict(
        self, X: Union[torch.Tensor, np.ndarray], raw_output: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Function to predict the output of the uncertainty given a dataloader.
        This function needs to be overwritten in the child classes.
        :param X: input data
        :param raw_output: Controls whether complete uncertainty information is returned or  only mean prediction and epistemic uncertainty
        :return: predictions
        """
        raise NotImplementedError(
            "This function needs to be overwritten in the child classes"
        )

    @torch.no_grad()
    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor, DataLoader],
        raw_output: bool = False,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        Function to predict the output of the uncertainty given an input.
        This acts as an wrapper of `_predict()`.
        :param X: input data
        :param raw_output: Controls whether complete uncertainty information is returned or  only mean prediction and epistemic uncertainty
        :return : predictions along with the uncertainty information if available
        """
        if isinstance(X, DataLoader):
            predictions, uncertainty = self.predict_routine_dataloader(X, raw_output)
        else:
            predictions, uncertainty = self.predict_routine_tensor(X, raw_output)

        if self.uncertainty_aware:
            return predictions, uncertainty
        return predictions

    def predict_routine_tensor(
        self, X: Union[torch.Tensor, np.ndarray], raw_output: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function to predict the output of the uncertainty given an input
        :param X: input data
        :param raw_output: Controls whether complete uncertainty information is returned or  only mean prediction and epistemic uncertainty
        :return: predictions, uncertainty (the uncertainty is -1 if not available)
        """
        X = validate_predict_input(X, self.device)
        predictions = self._predict(X, raw_output)
        if self.uncertainty_aware:
            return predictions[0], predictions[1]
        else:
            return predictions, torch.tensor([-1] * X.shape[0]).view(-1, 1).to(
                self.device
            )

    def predict_routine_dataloader(
        self, dataloader: DataLoader, raw_output: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function to predict the output of the uncertainty given a dataloader
        :param dataloader: dataloader object containing the input data
        :param raw_output: Controls whether complete uncertainty information is returned or  only mean prediction and epistemic uncertainty
        :return: predictions, uncertainty (the uncertainty is -1 if not available)
        """
        batch_size = dataloader.batch_size
        predictions = []
        uncertainty = []
        for inputs in iter(dataloader):
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs[0]

            # Every data instance is an input + label pair
            current_batch_size, remain = inputs.shape[0], inputs.shape[1:]

            # Pad the input to have same batch size
            inputs = torch.cat(
                [inputs, torch.zeros(batch_size - current_batch_size, *remain)],
                dim=0,
            )

            # move to device
            inputs = inputs.to(self.device)

            # Forward pass
            output = self._predict(inputs, raw_output)

            if self.uncertainty_aware:
                pred, unc = output

                # Reset padding
                pred = pred[:current_batch_size]
                unc = unc[:current_batch_size]

                predictions.append(pred)
                uncertainty.append(unc)
            else:
                pred = output

                # Reset padding
                pred = pred[:current_batch_size]

                predictions.append(pred)
                uncertainty.append(
                    torch.tensor([-1] * current_batch_size).view(-1, 1).to(self.device)
                )

        return torch.cat(predictions), torch.cat(uncertainty)

    def load(self, model_path: str) -> None:
        """
        Function to load the weights of deep learning uncertainty from a file.
        Sets uncertainty to eval mode
        :param model_path: complete file path to the module weights
        :return: Nothing
        """
        self.load_state_dict(torch.load(model_path, weights_only=True))
        self.eval()

    def save(self, model_path: str) -> None:
        """
        Function to save the weights of deep learning uncertainty to a file
        :param model_path: complete file path to the module weights
        :return: Nothing
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.state_dict(), model_path + f"{self.name}.pth")

    def init_optimizer_scheduler(
        self, optimizer_params: dict[str, str], scheduler_params: dict[str, str]
    ) -> None:
        """
        Function to initialize the optimizer and scheduler for the uncertainty.
        :param optimizer_params: dictionary containing the optimizer and its parameters in the field "optimizer" and "params" respectively
        :param scheduler_params: dictionary containing the scheduler and its parameters in the field "scheduler" and "params" respectively
        :return:
        """
        self.optimizer = optimizer_params["optimizer"](
            params=self.parameters(), **optimizer_params["params"]
        )
        self.scheduler = scheduler_params["scheduler"](
            self.optimizer, **scheduler_params["params"]
        )


class BaseEnsembleDNN(torch.nn.Module):
    """
    Base class for all ensemble deep learning models.
    This class contains the core training process and the predict function.
    It also contains the function to save and load the uncertainty weights.
    """

    def __init__(
        self,
        model: BaseDNN | list[BaseDNN],
        n_models: int,
        random_state: Optional[int] = None,
    ):
        """
        Constructor for the BaseEnsembleDNN class.
        :param model: The uncertainty to be used. This should be a torch.nn.Module object.
        :param n_models: number of models in the ensemble
        :param random_state: Random state for reproducibility. Default is None.
        """
        if random_state:
            set_seeds(random_state)

        super(BaseEnsembleDNN, self).__init__()

        # Will be set when called in fit
        self.current_epoch: Optional[int] = None
        self.n_batches: int = 0
        self.device: torch.device = torch.device("cpu")

        # General Information
        self.name: str = "BaseEnsembleDNN"
        self.uncertainty_aware: bool = True

        if isinstance(model, BaseDNN):
            self.ensemble = []
            for _ in range(n_models):
                ensemble_member = deepcopy(model)
                # Pertubate the weights of the ensemble, if only given one uncertainty
                for param in ensemble_member.model.parameters():
                    if param.requires_grad:
                        # Add Gaussian noise to the weight parameters
                        noise = torch.randn_like(param) * 0.01
                        param.data += noise
                self.ensemble.append(ensemble_member)

        elif isinstance(model, list):
            self.ensemble = deepcopy(model)
        else:
            raise ValueError("Model must be a BaseDNN or a list of BaseDNN models")

        # Decide on training routine
        self.training_routine = "ensemble"

        self.save_path = "models/"
        self.uncertainty_aware: bool = False

        # Saving the best uncertainty weights in validation
        self.best_model_weights = None

    def determine_training_routine(self):
        """
        Function to decide on the training routine of the ensemble
        :return: "slow_ensemble" or "fast_ensemble", depending on whether the ensemble contains batch normalization layers or not
        """
        if not any(
            [
                isinstance(module, torch.nn.BatchNorm1d)
                or isinstance(module, torch.nn.BatchNorm2d)
                or isinstance(module, torch.nn.BatchNorm3d)
                for module in self.ensemble[0].modules()
            ]
        ):
            return "fast_ensemble"
        return "slow_ensemble"

    def set_device(self, device: torch.device) -> None:
        """
        Function to set the device of the uncertainty
        :param device: device to set the uncertainty to
        :return: None
        """
        self.to(device)
        self.device = device
        [model.set_device(device) for model in self.ensemble]

    def load(self, model_path: str):
        """
        Loads the ensemble given a list of uncertainty paths.
        :param model_path: path to the folder containing ensemble models
        :return: None
        """
        for i, model in enumerate(self.ensemble):
            model.load_state_dict(
                torch.load(model_path + f"{model.name}_{i}.pth", weights_only=True)
            )
            model.train(False)
            model.to(self.device)

    def val_step(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Function to perform a single validation step for the ensemble
        :param inputs: The input data
        :param labels: The true labels
        :param kwargs: Additional parameters to pass to the validation step
        :return: The average loss of all models in the ensemble
        """
        self.eval()
        [model.eval() for model in self.ensemble]

        with torch.no_grad():
            loss = np.sum(
                [model.val_step(inputs, labels, **kwargs) for model in self.ensemble]
            )

        return loss

    def update_best_model(
        self, valloader: torch.utils.data.DataLoader, best_val_loss: float, epoch: int
    ) -> float:
        """
        Function to update the best uncertainty weights based on the validation loss

        :param valloader: dataloader containing the validation data
        :param best_val_loss: best validation loss so far
        :param epoch: current epoch number
        :param n_batches: amount of batches in the validation data
        :return: Best validation loss so far
        """
        # This self.eval() might be changed for MC Dropout.
        self.eval()
        [model.eval() for model in self.ensemble]

        n_batches = len(valloader)
        val_loss = 0

        for i, data in enumerate(valloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            val_loss += self.val_step(
                inputs,
                labels,
                **{"epoch": epoch, "iterpoch": i, "n_batches": n_batches},
            )
        print(
            f"In epoch:{epoch} average validation loss was Loss: {val_loss / n_batches}"
        )

        if val_loss < best_val_loss:
            self.best_model_weights = [
                deepcopy(model.state_dict()) for model in self.ensemble
            ]
            return val_loss
        return best_val_loss

    def _training_routine_slow(
        self,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader = None,
        optimizer_params: dict[str, int] = OPTIMIZER_CONFIGS["adam"],
        scheduler_params: dict[str, int] = SCHEDULER_CONFIGS["step"],
        n_epochs: int = 50,
    ):
        """
        Function to fit the deep learning models with a trainloader object
        :param trainloader: training data loader
        :param valloader: validation data loader
        :param optimizer_params: parameters to pass to the optimizer
        :param n_epochs: number of epochs
        :return: Saves the uncertainty to disk and returns None
        """
        for model in self.ensemble:
            model.set_device(self.device)
            model.compile(backend=BACKEND)
            model.init_optimizer_scheduler(optimizer_params, scheduler_params)

        for epoch in range(n_epochs):
            avg_loss = 0
            loaders = [iter(trainloader) for _ in range(len(self.ensemble))]
            for batch in range(len(trainloader)):
                # Get the individual data batches for the ensemble models

                for i, model in enumerate(self.ensemble):
                    input, target = next(loaders[i])

                    # Preparte input&target for uncertainty
                    input, target = pad_inputs(trainloader.batch_size, input, target)
                    input = input.to(self.device)
                    target = target.to(self.device)

                    # Make training step for uncertainty
                    avg_loss += model.train_step(input, target)

            avg_loss = avg_loss / len(trainloader)
            print(f"Finished Epoch {epoch} from {n_epochs} with {avg_loss}")

            for model in self.ensemble:
                model.scheduler.step()

        # Reset device to cpu
        self.set_device(torch.device("cpu"))

    def _training_routine_fast(
        self,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader = None,
        optimizer_params: dict[str, int] = OPTIMIZER_CONFIGS["adam"],
        scheduler_params: dict[str, int] = SCHEDULER_CONFIGS["step"],
        n_epochs: int = 50,
    ):
        """
        Function to fit the deep learning models with a trainloader object
        :param trainloader: training data loader
        :param valloader: validation data loader
        :param n_epochs: number of epochs
        :return: Saves the uncertainty to disk and returns None
        """

        batched_params, batched_buffer = stack_module_state(self.ensemble)
        self.optimizer = optimizer_params["optimizer"](
            params=batched_params.values(), **optimizer_params["params"]
        )
        self.scheduler = scheduler_params["scheduler"](
            self.optimizer, **scheduler_params["params"]
        )

        # Create a stateless meta uncertainty
        meta_model = deepcopy(self.ensemble[0]).to("meta")
        meta_model.compile()

        def fmodel(params, buffers, x):
            return functional_call(meta_model, (params, buffers), (x,))

        for epoch in range(n_epochs):
            avg_loss = 0
            loaders = [iter(trainloader) for _ in range(len(self.ensemble))]
            for batch in range(len(trainloader)):
                # Get the individual data batches for the ensemble models
                data = [next(loader) for loader in loaders]

                # gather the inputs and targets and stack for parallel application
                input, target = zip(*data)
                input = torch.stack(input).to(self.device)
                target = torch.stack(target).to(self.device)

                # Parallel forward pass of all ensembles
                pred = vmap(fmodel, randomness="same")(
                    batched_params, batched_buffer, input
                )

                # Compute the loss of all ensembles
                partial_loss_function = functools.partial(
                    self.ensemble[0].loss_function,
                    **{
                        "iterpoch": batch,
                        "epoch": epoch,
                        "n_batches": len(trainloader),
                    },
                )
                loss = (
                    vmap(partial_loss_function, randomness="same")(pred, target)
                ).sum()
                avg_loss += loss

                # Update the params of the ensembles parallel
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            avg_loss = avg_loss / len(trainloader)
            print(f"Finished Epoch {epoch} from {n_epochs} with {avg_loss}")

            # Adjust learning rate
            self.scheduler.step()

        # Reset device to cpu
        self.set_device(torch.device("cpu"))

        # As the optimizer updates self.params, they are moved back into the models
        for i, model in enumerate(self.ensemble):
            state_dict = model.state_dict()
            for name, param in batched_params.items():
                state_dict[name] = param[i]
            model.load_state_dict(state_dict)

    def save(self, model_path: str) -> None:
        """
        Function to save the weights of deep learning uncertainty to a folder path
        :param model_path: complete path to the folder in which the models will be saved
        :return: Nothing
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i, model in enumerate(self.ensemble):
            torch.save(
                model.state_dict(), model_path + "{}_{}.pth".format(model.name, i)
            )

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
        Function to fit the deep learning uncertainty with a trainloader object.
        This function will also save the uncertainty to disk if a dataset name is provided.

        :param X_train: features or complete data loader
        :param Y_train: labels or None
        :param X_val: features or complete data loader
        :param Y_val: labels or None
        :param dataset_name: name of the dataset to save the uncertainty. If None the uncertainty will not be saved to an folder, otherwise it will be saved to the path models/dataset_name/
        :param n_epochs: number of epochs
        :param batch_size: batch size for the dataloader
        :param optimizer_params: dictionary containing the optimizer and its parameters in the field "optimizer" and "params" respectively. See `config.py` for more information.
        :param scheduler_params: dictionary containing the scheduler and its parameters in the field "scheduler" and "params" respectively. See `config.py` for more information.
        :return: self
        """

        self.set_device(find_device())

        trainloader = _convert_to_dataloader(X_train, Y_train, batch_size)
        valloader = _convert_to_dataloader(X_val, Y_val, batch_size)

        if self.determine_training_routine() == "fast_ensemble":
            print("Using fast routine")

            self._training_routine_fast(
                trainloader,
                valloader,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                n_epochs=n_epochs,
            )
        else:
            warnings.warn(
                'As your base_models contain Batchnormalisation we switch to "squentiell" training of each ensemble member. This is not a problem for small models but may be bad for bigger models.'
                "One suggestion is to remove Batchnorm or just train each uncertainty independently and then arange the saved weights in a folder and load them with .load()"
            )
            self._training_routine_slow(
                trainloader,
                valloader,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                n_epochs=n_epochs,
            )

        # Reset current epoch
        self.current_epoch = None

        if dataset_name is not None:
            self.save(self.save_path + dataset_name + "/" + self.name + "/")

        return self

    @torch.no_grad()
    def _predict(
        self, X: Union[torch.Tensor, np.ndarray], raw_output: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Function to predict the output of the uncertainty given X.
        This function needs to be overwritten in the child classes.
        :param X: input data
        :param raw_output: Controls whether complete uncertainty information is returned or  only mean prediction and epistemic uncertainty
        :return: predictions
        """
        raise NotImplementedError(
            "This function needs to be overwritten in the child classes"
        )

    @torch.no_grad()
    def predict(
        self, X: Union[np.ndarray, torch.Tensor], raw_output: bool = False
    ) -> Any:
        """
        Function to predict the output of the uncertainty given an input
        :param X:
        :return:
        """
        predictions = []
        uncertainty = []
        DATALOADER_CONFIGS["shuffle"] = False
        if isinstance(X, DataLoader):
            batch_size = X.batch_size

            for inputs in iter(X):
                if not isinstance(inputs, torch.Tensor):
                    inputs = inputs[0]

                # Every data instance is an input + label pair
                current_batch_size, remain = inputs.shape[0], inputs.shape[1:]

                # Pad the input to have same batch size
                inputs = torch.cat(
                    [inputs, torch.zeros(batch_size - current_batch_size, *remain)],
                    dim=0,
                )

                inputs = inputs.to(self.device)

                # output.shape == (n_models, n_samples, n_outputs, 1)
                output = self._predict(inputs, raw_output)

                if self.uncertainty_aware:
                    pred, uncer = output

                    pred = pred[:current_batch_size]
                    uncer = uncer[:current_batch_size]

                    predictions.append(pred)
                    uncertainty.append(uncer)
                else:
                    pred = predictions

                    pred = pred[:current_batch_size]

                    predictions.append(pred)
        else:
            # output.shape == (n_models, n_samples, n_outputs, 1)
            output = self._predict(X, raw_output)
            if self.uncertainty_aware:
                pred, unc = output

                predictions.append(pred)
                uncertainty.append(unc)
            else:
                pred = output

                predictions.append(pred)

        if self.uncertainty_aware:
            return torch.cat(predictions), torch.cat(uncertainty)
        DATALOADER_CONFIGS["shuffle"] = True
        return torch.cat(predictions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs the \mu and \sigma of the individual ensemble members.
        :param x: input data
        :return:\mu and \sigma of individual ensemble members
        """
        preds = torch.stack([model(x) for model in self.ensemble], dim=1)

        return preds

    def eval(self: T) -> T:
        """
        Function to set the uncertainty to evaluation mode
        :return: self
        """
        [model.eval() for model in self.ensemble]
        return self


def get_pretrained(
    model_name: str, **kwargs
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Function to load a pre-trained uncertainty from the torch hub
    :param  model_name: name of the uncertainty to load
    :param  kwargs: additional parameters to pass to the uncertainty
    :return: torch.nn.Module
    """
    if model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1
        preprocessor = weights.transforms()
        return resnet50(weights=weights, **kwargs), preprocessor
    elif model_name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        preprocessor = weights.transforms()
        return resnet18(weights="DEFAULT", **kwargs), preprocessor
    elif model_name == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights

        weights = VGG16_Weights.IMAGENET1K_V1
        preprocessor = weights.transforms()
        return vgg16(weights="DEFAULT", **kwargs), preprocessor
    elif model_name == "efficientnet_b3":
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        preprocessor = weights.transforms()
        return efficientnet_b3(weights="DEFAULT", **kwargs), preprocessor
    else:
        raise ValueError(
            "Model name not recognized. Until now only resnet50, resnet16, vgg16 and efficientnet_b3 are supported."
        )


def pretrained_to_basednn(model_name: str, **kwargs) -> BaseDNN:
    """
    Function to load a pre-trained uncertainty from the torch hub and convert it to a BaseDNN uncertainty
    :param  model_name: name of the uncertainty to load
    :param  kwargs: additional parameters to pass to the uncertainty
    :return: BaseDNN
    """
    model, preprocessor = get_pretrained(model_name, **kwargs)
    model = BaseDNN(model)
    model.name = model_name
    return model

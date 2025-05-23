"""
This file contains the implementation of the models used for classification tasks.
"""
from typing import Optional

import torch
from epiuc.uncertainty.base import BaseDNN, pretrained_to_basednn
from epiuc.utils.general import (
    set_seeds,
    validate_mlp_params,
    construct_mlp,
    set_model_to_eval,
    validate_predict_input,
)


def _calculate_LeNet_maxpool_padding(image_dim):
    # Assumes image_width == image_height
    # Also assumes stride = 2, kernel = 2
    return image_dim - (image_dim // 2)


def _standard_class_prediction(model, X, raw_output: bool = False):
    """
    This method is used to get the predicted probability of the uncertainty.
    It uses the softmax function to convert the logits to probabilities.
    :param model: The uncertainty to be used for prediction.
    :param X: The input data.
    :param raw_output: If True, return the logits of the uncertainty, else return the probability.
    :return: The probability of the uncertainty or logits depending on the raw_output parameter.
    """
    set_model_to_eval(model)

    X = validate_predict_input(X, model.device)

    logits = model(X)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if raw_output:
        return logits
    return probs


class Classifier(BaseDNN):
    def __init__(self, model, random_state=None):
        super().__init__(model=model, random_state=random_state)
        self.name = "Classifier"
        self.uncertainty_aware = False

    def loss_function(self, output, target, **kwargs):
        """
        Loss function for the classification task. It uses the cross entropy loss function.
        :param output: The output of the uncertainty.
        :param target: The target labels.
        :param kwargs: Additional arguments for the loss function. not used.
        :return: The cross entropy loss.
        """
        cross_entropy_loss = torch.nn.functional.cross_entropy(
            input=output, target=target
        )

        return torch.mean(cross_entropy_loss)

    def _predict(self, X, raw_output: bool = False):
        """
        This method is used to get the predicted probability of the uncertainty.
        :param X: input data
        :param raw_output: If True, return the logits of the uncertainty, else return the probability.
        :return: The probability of the uncertainty or logits depending on the raw_output parameter.
        """
        return _standard_class_prediction(self, X, raw_output=raw_output)


class MLP_Classifier(Classifier):
    def __init__(
        self,
        input_shape: int,
        n_classes: int,
        n_layers: int,
        num_neurons: int,
        dropout_prob: float = 0.5,
        batch_norm: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Simple MLP Classifier.
        :param input_shape: The number of input features.
        :param n_classes: The number of classes.
        :param n_layers: The number of layers.
        :param num_neurons: The number of neurons in each layer.
        :param dropout_prob: The dropout rate.
        :param batch_norm: If True, apply batch normalization after each layer.
        :param random_state: The random state for reproducibility.
        """
        if random_state:
            set_seeds(random_state)

        validate_mlp_params(n_layers, num_neurons, dropout_prob, input_shape, n_classes)

        model = construct_mlp(
            input_shape, n_classes, n_layers, num_neurons, dropout_prob, batch_norm
        )

        super().__init__(model=model, random_state=random_state)

        self.name = "MLP_Classifier"


class LeNet_MNIST(Classifier):
    def __init__(
        self,
        in_channels,
        image_width,
        image_heigth,
        drop_prob=0.5,
        random_state: Optional[int] = None,
    ):
        """
        LeNet uncertainty for MNIST image classification.
        :param in_channels: The number of input channels.
        :param image_width: The width of the input image.
        :param image_heigth: The height of the input image.
        :param drop_prob: The dropout probability.
        :param random_state: The random state for reproducibility.
        """

        if image_width != image_heigth:
            raise ValueError("Model not yet tested for non quadratic images!")

        padding = _calculate_LeNet_maxpool_padding(image_width)

        ## Model Creation START ##
        # Feature Extraction Layers
        conv1_out_channels = 20
        conv2_out_channels = 50

        features = torch.nn.Sequential(
            # 1 Layer: CNN -> ReLU -> Padding -> MaxPool
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv1_out_channels,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding="same",
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.ConstantPad2d(padding, value=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            # 2 Layer: CNN -> ReLU -> Padding -> MaxPool
            torch.nn.Conv2d(
                in_channels=in_channels * conv1_out_channels,
                out_channels=conv2_out_channels,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding="same",
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.ConstantPad2d(padding, value=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

        logits = torch.nn.Sequential(
            torch.nn.Flatten(),
            # 3 Layer: Fully Connected -> ReLU
            torch.nn.Linear(
                in_features=image_width * image_heigth * conv2_out_channels,
                out_features=500,
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_prob),
            # 4 Layer: Fully Connected ( Output Layer)
            torch.nn.Linear(in_features=500, out_features=10, bias=True),
        )
        super(LeNet_MNIST, self).__init__(
            model=torch.nn.Sequential(features, logits),
            random_state=random_state,
        )

        ## Model Creation END ##

        self.name = "LeNet_MNIST"
        self.uncertainty_aware = False

        # Regularization
        self.regularization_params = [*logits.parameters()]


class Resnet18_Imagenet(Classifier):
    """
    Resnet18 uncertainty for Imagenet classification.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Loads pretrained Resnet18 uncertainty for Imagenet image classification.
        """
        model = pretrained_to_basednn("resnet18")

        super().__init__(model=model, random_state=random_state)

        self.name = "Resnet18_Imagenet"


class Resnet18_Cifar10(Classifier):
    """
    Resnet18 uncertainty for Cifar10 classification.
    """

    def __init__(self, random_state, pretrained=False):
        """
        Resnet18 uncertainty for Cifar10 classification.
        :param random_state: The random state for reproducibility.
        :param pretrained: If True, load the pretrained uncertainty.
        """
        from epiuc.uncertainty.pytorch_cifar_models import cifar10_resnet20

        model = cifar10_resnet20(pretrained=pretrained, progress=True)

        super().__init__(
            model=torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.layer1,
                model.layer2,
                model.layer3,
                model.avgpool,
                torch.nn.Flatten(start_dim=1),
                model.fc,
            ),
            random_state=random_state,
        )

        self.name = "Resnet18_Cifar10"


class Resnet50_Imagenet(Classifier):
    """
    Resnet50 uncertainty for Imagenet classification.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Loads pretrained Resnet50 uncertainty for Imagenet image classification.
        :param random_state:
        """
        model = pretrained_to_basednn("resnet50")
        super().__init__(model=model, random_state=random_state)
        self.name = "Resnet50_Imagenet"


class Resnet50_Cifar10(Classifier):
    """
    Resnet50 uncertainty for Cifar10 classification.
    """

    def __init__(self, random_state, pretrained=False):
        """
        Resnet50 uncertainty for Cifar10 classification.

        :param random_state: The random state for reproducibility.
        :param pretrained: If True, load the pretrained uncertainty.
        """
        from epiuc.uncertainty.pytorch_cifar_models import cifar10_resnet56

        model = cifar10_resnet56(pretrained=pretrained, progress=True)

        super().__init__(
            model=model,
            random_state=random_state,
        )

        self.name = "Resnet50_Cifar10"


class Resnet18_Cifar100(Classifier):
    """
    Resnet18 uncertainty for Cifar100 classification.
    """

    def __init__(self, random_state, pretrained=False):
        """
        Resnet18 uncertainty for Cifar100 classification.

        :param random_state: The random state for reproducibility.
        :param pretrained: If True, load the pretrained uncertainty.
        """
        from epiuc.uncertainty.pytorch_cifar_models import cifar100_resnet20

        model = cifar100_resnet20(pretrained=pretrained, progress=True)

        super().__init__(
            model=model,
            random_state=random_state,
        )

        self.name = "Resnet18_Cifar100"


class Resnet50_Cifar100(Classifier):
    """
    Resnet50 uncertainty for Cifar100 classification.
    """

    def __init__(self, random_state, pretrained=False):
        """
        Resnet50 uncertainty for Cifar100 classification.
        :param random_state: The random state for reproducibility.
        :param pretrained:  If True, load the pretrained uncertainty.
        """
        from epiuc.uncertainty.pytorch_cifar_models import cifar100_resnet56

        model = cifar100_resnet56(pretrained=pretrained, progress=True)

        super().__init__(
            model=model,
            random_state=random_state,
        )

        self.name = "Resnet50_Cifar100"


class VGG16_Imagenet(Classifier):
    """
    VGG16 uncertainty for Imagenet classification.
    """

    def __init__(self, random_state):
        """
        Loads pretrained VGG16 uncertainty for Imagenet image classification.
        :param random_state: The random state for reproducibility.
        """
        model = pretrained_to_basednn("vgg16")

        super().__init__(model=model, random_state=random_state)

        self.name = model.name

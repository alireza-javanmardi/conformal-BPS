import os
from functools import partial

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import h5py

from epiuc.CustomDataset import CustomDataset
from epiuc.config import (
    DATALOADER_CONFIGS,
    IMAGENET_C_PERTUBATIONS,
    CIFAR_C_PERTUBATIONS,
    ROOT_PATH,
    BASE_PATH,
)
from epiuc.utils.general import seed_worker, min_max_normalize


##############################
####    Image Datasets   #####
##############################


def _build_val_test_loaders(dataset, vali_size, generator, loading_configs):
    # Calculate sizes for training and validation sets
    test_size = int((1 - vali_size) * len(dataset))
    val_size = len(dataset) - test_size

    # Split dataset into training and validation sets
    if val_size > 0:
        if generator is not None:
            testset, valset = torch.utils.data.random_split(
                dataset,
                [test_size, val_size],
                generator=generator,
            )
        else:
            testset, valset = torch.utils.data.random_split(
                dataset, [test_size, val_size]
            )

        testset = torch.utils.data.Subset(dataset, testset.indices)
        valloader = torch.utils.data.Subset(dataset, valset.indices)
    else:
        testset = dataset
        valloader = None

    testloader = torch.utils.data.DataLoader(
        testset,
        **loading_configs,
    )

    return valloader, testloader


def load_image_data(
    dataset_name, vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS
):
    """
    Load image dataset
    :param dataset_name: name of image dataset
    :param vali_size: proportion of validation set
    :param generator: random number generator (relevant for reproducibility). Defaults to None, implying the reproducibility is not guaranteed.
    :param loading_configs: configurations for DataLoader should include (num_workers, pin_memory, batch_size, shuffle). Defaults to IMAGE_DATALOADER_CONFIG (epiuc/config.py)
    :return: trainloader, valloader, testloader
    """
    if dataset_name == "mnist":
        return load_mnist(vali_size, generator, loading_configs)
    elif dataset_name == "cifar10":
        return load_cifar10(vali_size, generator, loading_configs)
    elif dataset_name == "cifar100":
        return load_cifar100(vali_size, generator, loading_configs)
    elif dataset_name == "imagenet":
        return load_imagenet(vali_size, generator, loading_configs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_mnist(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS):
    """
    Load MNIST dataset.
    See `load_image_data` for more details on the arguments.
    :return: MNIST trainloader, valloader, testloader
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )

    trainset = torchvision.datasets.MNIST(
        root=ROOT_PATH, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, **loading_configs, worker_init_fn=seed_worker
    )

    testset = torchvision.datasets.MNIST(
        root=ROOT_PATH, train=False, download=True, transform=transform
    )
    valloader, testloader = _build_val_test_loaders(
        testset, vali_size, generator, loading_configs
    )

    return trainloader, valloader, testloader


def load_cifar10(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS):
    """
    Load CIFAR-10 dataset.
    See `load_image_data` for more details on the arguments.
    :return: CIFAR10 trainloader, valloader, testloader
    """
    # Transformation pipeline
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
        ]
    )

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=ROOT_PATH, train=True, download=True, transform=transform
    )

    # Create DataLoaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(
        trainset,
        **DATALOADER_CONFIGS,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
        ]
    )
    # Load CIFAR-10 tests dataset
    testset = torchvision.datasets.CIFAR10(
        root=ROOT_PATH, train=False, download=True, transform=transform
    )

    valloader, testloader = _build_val_test_loaders(
        testset, vali_size, generator, loading_configs
    )

    return trainloader, valloader, testloader


def load_cifar100(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS):
    """
    Load CIFAR-100 dataset.
    See `load_image_data` for more details on the arguments.
    :return: CIFAR100 trainloader, valloader, testloader
    """

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ]
    )
    trainset = torchvision.datasets.CIFAR100(
        root=ROOT_PATH, train=True, download=True, transform=transform
    )
    # Create DataLoaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(
        trainset,
        **DATALOADER_CONFIGS,
        worker_init_fn=seed_worker,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ]
    )

    testset = torchvision.datasets.CIFAR100(
        root=ROOT_PATH, train=False, download=True, transform=transform
    )

    valloader, testloader = _build_val_test_loaders(
        testset, vali_size, generator, loading_configs
    )

    return trainloader, valloader, testloader


def load_imagenet(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS):
    """
    Load ImageNet dataset.
    Only works in  LRZ Cluster Enviroment.
    See `load_image_data` for more details on the arguments.
    :return: IMAGENET trainloader, valloader, testloader
    """
    print("Loading ImageNet only works on Cluser")
    path = os.path.join(os.environ["SCRATCH"], "ILSVRC/Data/CLS-LOC")
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        train_path,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    # Create DataLoaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        **DATALOADER_CONFIGS,
    )

    testset = torchvision.datasets.ImageFolder(
        test_path,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valloader, testloader = _build_val_test_loaders(
        testset, vali_size, generator, loading_configs
    )

    return trainloader, valloader, testloader


def load_imagenet_local(
    vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS
):
    """
    Loads only Imagenet tests data..
    :return: IMAGENET valloader, testloader
    """
    transformation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    testset = (
        torchvision.datasets.ImageFolder(ROOT_PATH + "imagenet-val", transformation),
    )

    valloader, testloader = _build_val_test_loaders(
        testset, vali_size, generator, loading_configs
    )

    return valloader, testloader


def load_cifar10_c(pertubation, severity, loading_configs=DATALOADER_CONFIGS):
    """
    Load CIFAR-10-C dataset.

    :param pertubation: Which pertubation to load. See CIFAR_C_PERTUBATIONS for possible values (epiuc/config.py).
    :param severity: The degree of pertubation. Must be one of {1,2,3,4,5}.
    :param loading_configs: DataLoader configurations. Defaults to IMAGE_DATALOADER_CONFIG (epiuc/config.py).
    :return: clean_val_loader, pertubated_val_loader of CIFAR10
    """
    if not (1 <= severity <= 5):
        raise ValueError("Wrong seversity value. Must be one of {1,2,3,4,5}")

    if pertubation not in CIFAR_C_PERTUBATIONS:
        raise ValueError("Wrong pertubation")

    # Reset shuffling to align the datapoints
    loading_configs["shuffle"] = False

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
        ]
    )

    data = np.load(ROOT_PATH + f"CIFAR-10-C/{pertubation}.npy")[
        (severity - 1) * 10000 : severity * 10000
    ]
    labels = np.load(ROOT_PATH + "CIFAR-10-C/labels.npy")[
        (severity - 1) * 10000 : severity * 10000
    ]

    dataset = CustomDataset(X=data, Y=labels, transform=transform)

    clean_val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=ROOT_PATH, train=False, download=True, transform=transform
        ),
        **loading_configs,
        worker_init_fn=seed_worker,
    )

    pertubated_val_loader = torch.utils.data.DataLoader(
        dataset,
        **loading_configs,
    )

    return clean_val_loader, pertubated_val_loader


def load_cifar10_h(loading_configs=DATALOADER_CONFIGS):
    """
    Load CIFAR-10-H dataset along CIFAR10 Test set
    :param loading_configs: DataLoader configurations. Defaults to DATALOADER_CONFIGS (epiuc/config.py).
    :return: cifar10_loader, cifar_10_h_loader
    """
    loading_configs["shuffle"] = False

    data = np.load(ROOT_PATH + "CIFAR-10-H/cifar10h-probs.npy")
    labels = np.argmax(data, axis=1)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
        ]
    )

    cifar_10_h = CustomDataset(X=data, Y=labels)

    cifar_10_h_loader = torch.utils.data.DataLoader(
        cifar_10_h, **loading_configs, worker_init_fn=seed_worker
    )

    cifar10_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=ROOT_PATH, train=False, download=True, transform=transform
        ),
        **loading_configs,
        worker_init_fn=seed_worker,
    )

    return cifar10_loader, cifar_10_h_loader


def load_cifar100_c(pertubation, severity, loading_configs=DATALOADER_CONFIGS):
    """
    Load CIFAR-100-C dataset.

    :param pertubation: Which pertubation to load. See CIFAR_C_PERTUBATIONS for possible values (epiuc/config.py).
    :param severity: The degree of pertubation. Must be one of {1,2,3,4,5}.
    :param loading_configs: DataLoader configurations. Defaults to IMAGE_DATALOADER_CONFIG (epiuc/config.py).
    :return: clean_val_loader, pertubated_val_loader of CIFAR100
    """
    if not (1 <= severity <= 5):
        raise ValueError("Wrong seversity value. Must be one of {1,2,3,4,5}")

    if pertubation not in CIFAR_C_PERTUBATIONS:
        raise ValueError("Wrong pertubation")

    # Reset shuffling to align the datapoints
    loading_configs["shuffle"] = False

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ]
    )

    data = np.load(ROOT_PATH + f"CIFAR-100-C/{pertubation}.npy")[
        (severity - 1) * 10000 : severity * 10000
    ]
    labels = np.load(ROOT_PATH + "CIFAR-100-C/labels.npy")[
        (severity - 1) * 10000 : severity * 10000
    ]

    dataset = CustomDataset(X=data, Y=labels, transform=transform)

    clean_val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            root=ROOT_PATH, train=False, download=True, transform=transform
        ),
        **loading_configs,
    )

    pertubated_val_loader = torch.utils.data.DataLoader(
        dataset,
        **loading_configs,
    )

    return clean_val_loader, pertubated_val_loader


def load_imagenet_c(
    pertubation, severity, generator=None, loading_configs=DATALOADER_CONFIGS
):
    """
    Load  Imagenet dataset.

    :param pertubation: Which pertubation to load. See IMAGENET_C_PERTUBATIONS for possible values (epiuc/config.py).
    :param severity: The degree of pertubation. Must be one of {1,2,3,4,5}.
    :param loading_configs: DataLoader configurations. Defaults to IMAGE_DATALOADER_CONFIG (epiuc/config.py).
    :return: clean_val_loader, pertubated_val_loader of Imagenet
    """
    if not (1 <= severity <= 5):
        raise ValueError("Wrong seversity value. Must be one of {1,2,3,4,5}")

    if pertubation not in IMAGENET_C_PERTUBATIONS:
        raise ValueError("Wrong ")

    print("Loading ImageNet Fog only works on Cluster")
    pertubated_path = os.path.join(os.environ["SCRATCH"], f"{pertubation}/{severity}")
    clean_path = os.path.join(os.environ["SCRATCH"], "ILSVRC/Data/CLS-LOC/val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    clean_val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            clean_path,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        **DATALOADER_CONFIGS,
    )

    pertubated_val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            pertubated_path,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        **DATALOADER_CONFIGS,
    )

    return clean_val_loader, pertubated_val_loader


##############################
###  Regression Datasets  ####
##############################


def load_cubic(x_min, x_max, n):
    x_train = np.linspace(x_min, x_max, n)
    x_test = np.linspace(x_min - 3, x_max + 3, n)

    x_train = np.expand_dims(x_train, -1).astype(np.float32)
    x_test = np.expand_dims(x_test, -1).astype(np.float32)

    sigma_train = 3 * np.ones_like(x_train)
    sigma_text = np.zeros_like(x_test)

    y_train = x_train**3 + np.random.normal(0, sigma_train).astype(np.float32)
    y_test = x_test**3 + np.random.normal(0, sigma_text).astype(np.float32)

    return x_train, y_train, x_test, y_test


def load_star():
    df = pd.read_csv(BASE_PATH + "STAR.csv")
    df.loc[df["gender"] == "female", "gender"] = 0
    df.loc[df["gender"] == "male", "gender"] = 1

    df.loc[df["ethnicity"] == "cauc", "ethnicity"] = 0
    df.loc[df["ethnicity"] == "afam", "ethnicity"] = 1
    df.loc[df["ethnicity"] == "asian", "ethnicity"] = 2
    df.loc[df["ethnicity"] == "hispanic", "ethnicity"] = 3
    df.loc[df["ethnicity"] == "amindian", "ethnicity"] = 4
    df.loc[df["ethnicity"] == "other", "ethnicity"] = 5

    df.loc[df["stark"] == "regular", "stark"] = 0
    df.loc[df["stark"] == "small", "stark"] = 1
    df.loc[df["stark"] == "regular+aide", "stark"] = 2

    df.loc[df["star1"] == "regular", "star1"] = 0
    df.loc[df["star1"] == "small", "star1"] = 1
    df.loc[df["star1"] == "regular+aide", "star1"] = 2

    df.loc[df["star2"] == "regular", "star2"] = 0
    df.loc[df["star2"] == "small", "star2"] = 1
    df.loc[df["star2"] == "regular+aide", "star2"] = 2

    df.loc[df["star3"] == "regular", "star3"] = 0
    df.loc[df["star3"] == "small", "star3"] = 1
    df.loc[df["star3"] == "regular+aide", "star3"] = 2

    df.loc[df["lunchk"] == "free", "lunchk"] = 0
    df.loc[df["lunchk"] == "non-free", "lunchk"] = 1

    df.loc[df["lunch1"] == "free", "lunch1"] = 0
    df.loc[df["lunch1"] == "non-free", "lunch1"] = 1

    df.loc[df["lunch2"] == "free", "lunch2"] = 0
    df.loc[df["lunch2"] == "non-free", "lunch2"] = 1

    df.loc[df["lunch3"] == "free", "lunch3"] = 0
    df.loc[df["lunch3"] == "non-free", "lunch3"] = 1

    df.loc[df["schoolk"] == "inner-city", "schoolk"] = 0
    df.loc[df["schoolk"] == "suburban", "schoolk"] = 1
    df.loc[df["schoolk"] == "rural", "schoolk"] = 2
    df.loc[df["schoolk"] == "urban", "schoolk"] = 3

    df.loc[df["school1"] == "inner-city", "school1"] = 0
    df.loc[df["school1"] == "suburban", "school1"] = 1
    df.loc[df["school1"] == "rural", "school1"] = 2
    df.loc[df["school1"] == "urban", "school1"] = 3

    df.loc[df["school2"] == "inner-city", "school2"] = 0
    df.loc[df["school2"] == "suburban", "school2"] = 1
    df.loc[df["school2"] == "rural", "school2"] = 2
    df.loc[df["school2"] == "urban", "school2"] = 3

    df.loc[df["school3"] == "inner-city", "school3"] = 0
    df.loc[df["school3"] == "suburban", "school3"] = 1
    df.loc[df["school3"] == "rural", "school3"] = 2
    df.loc[df["school3"] == "urban", "school3"] = 3

    df.loc[df["degreek"] == "bachelor", "degreek"] = 0
    df.loc[df["degreek"] == "master", "degreek"] = 1
    df.loc[df["degreek"] == "specialist", "degreek"] = 2
    df.loc[df["degreek"] == "master+", "degreek"] = 3

    df.loc[df["degree1"] == "bachelor", "degree1"] = 0
    df.loc[df["degree1"] == "master", "degree1"] = 1
    df.loc[df["degree1"] == "specialist", "degree1"] = 2
    df.loc[df["degree1"] == "phd", "degree1"] = 3

    df.loc[df["degree2"] == "bachelor", "degree2"] = 0
    df.loc[df["degree2"] == "master", "degree2"] = 1
    df.loc[df["degree2"] == "specialist", "degree2"] = 2
    df.loc[df["degree2"] == "phd", "degree2"] = 3

    df.loc[df["degree3"] == "bachelor", "degree3"] = 0
    df.loc[df["degree3"] == "master", "degree3"] = 1
    df.loc[df["degree3"] == "specialist", "degree3"] = 2
    df.loc[df["degree3"] == "phd", "degree3"] = 3

    df.loc[df["ladderk"] == "level1", "ladderk"] = 0
    df.loc[df["ladderk"] == "level2", "ladderk"] = 1
    df.loc[df["ladderk"] == "level3", "ladderk"] = 2
    df.loc[df["ladderk"] == "apprentice", "ladderk"] = 3
    df.loc[df["ladderk"] == "probation", "ladderk"] = 4
    df.loc[df["ladderk"] == "pending", "ladderk"] = 5
    df.loc[df["ladderk"] == "notladder", "ladderk"] = 6

    df.loc[df["ladder1"] == "level1", "ladder1"] = 0
    df.loc[df["ladder1"] == "level2", "ladder1"] = 1
    df.loc[df["ladder1"] == "level3", "ladder1"] = 2
    df.loc[df["ladder1"] == "apprentice", "ladder1"] = 3
    df.loc[df["ladder1"] == "probation", "ladder1"] = 4
    df.loc[df["ladder1"] == "noladder", "ladder1"] = 5
    df.loc[df["ladder1"] == "notladder", "ladder1"] = 6

    df.loc[df["ladder2"] == "level1", "ladder2"] = 0
    df.loc[df["ladder2"] == "level2", "ladder2"] = 1
    df.loc[df["ladder2"] == "level3", "ladder2"] = 2
    df.loc[df["ladder2"] == "apprentice", "ladder2"] = 3
    df.loc[df["ladder2"] == "probation", "ladder2"] = 4
    df.loc[df["ladder2"] == "noladder", "ladder2"] = 5
    df.loc[df["ladder2"] == "notladder", "ladder2"] = 6

    df.loc[df["ladder3"] == "level1", "ladder3"] = 0
    df.loc[df["ladder3"] == "level2", "ladder3"] = 1
    df.loc[df["ladder3"] == "level3", "ladder3"] = 2
    df.loc[df["ladder3"] == "apprentice", "ladder3"] = 3
    df.loc[df["ladder3"] == "probation", "ladder3"] = 4
    df.loc[df["ladder3"] == "noladder", "ladder3"] = 5
    df.loc[df["ladder3"] == "notladder", "ladder3"] = 6

    df.loc[df["tethnicityk"] == "cauc", "tethnicityk"] = 0
    df.loc[df["tethnicityk"] == "afam", "tethnicityk"] = 1

    df.loc[df["tethnicity1"] == "cauc", "tethnicity1"] = 0
    df.loc[df["tethnicity1"] == "afam", "tethnicity1"] = 1

    df.loc[df["tethnicity2"] == "cauc", "tethnicity2"] = 0
    df.loc[df["tethnicity2"] == "afam", "tethnicity2"] = 1

    df.loc[df["tethnicity3"] == "cauc", "tethnicity3"] = 0
    df.loc[df["tethnicity3"] == "afam", "tethnicity3"] = 1
    df.loc[df["tethnicity3"] == "asian", "tethnicity3"] = 2

    df = df.dropna()

    grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
    grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]

    names = df.columns
    # Optionally have target_names
    # target_names = names[8:16]
    data_names = np.concatenate((names[0:8], names[17:]))
    X = df.loc[:, data_names].values.astype(np.float32)
    y = grade.values.astype(np.float32)
    return X, y


def load_bio():
    # https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
    df = pd.read_csv(BASE_PATH + "CASP.csv")
    y = df.iloc[:, 0].values.astype(np.float32)
    X = df.iloc[:, 1:].values.astype(np.float32)
    return X, y


def load_meps19():
    df = pd.read_csv(BASE_PATH + "meps_19_reg.csv")
    column_names = df.columns
    response_name = "UTILIZATION_reg"
    column_names = column_names[column_names != response_name]
    column_names = column_names[column_names != "Unnamed: 0"]

    col_names = [
        "AGE",
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PERWT15F",
        "REGION=1",
        "REGION=2",
        "REGION=3",
        "REGION=4",
        "SEX=1",
        "SEX=2",
        "MARRY=1",
        "MARRY=2",
        "MARRY=3",
        "MARRY=4",
        "MARRY=5",
        "MARRY=6",
        "MARRY=7",
        "MARRY=8",
        "MARRY=9",
        "MARRY=10",
        "FTSTU=-1",
        "FTSTU=1",
        "FTSTU=2",
        "FTSTU=3",
        "ACTDTY=1",
        "ACTDTY=2",
        "ACTDTY=3",
        "ACTDTY=4",
        "HONRDC=1",
        "HONRDC=2",
        "HONRDC=3",
        "HONRDC=4",
        "RTHLTH=-1",
        "RTHLTH=1",
        "RTHLTH=2",
        "RTHLTH=3",
        "RTHLTH=4",
        "RTHLTH=5",
        "MNHLTH=-1",
        "MNHLTH=1",
        "MNHLTH=2",
        "MNHLTH=3",
        "MNHLTH=4",
        "MNHLTH=5",
        "HIBPDX=-1",
        "HIBPDX=1",
        "HIBPDX=2",
        "CHDDX=-1",
        "CHDDX=1",
        "CHDDX=2",
        "ANGIDX=-1",
        "ANGIDX=1",
        "ANGIDX=2",
        "MIDX=-1",
        "MIDX=1",
        "MIDX=2",
        "OHRTDX=-1",
        "OHRTDX=1",
        "OHRTDX=2",
        "STRKDX=-1",
        "STRKDX=1",
        "STRKDX=2",
        "EMPHDX=-1",
        "EMPHDX=1",
        "EMPHDX=2",
        "CHBRON=-1",
        "CHBRON=1",
        "CHBRON=2",
        "CHOLDX=-1",
        "CHOLDX=1",
        "CHOLDX=2",
        "CANCERDX=-1",
        "CANCERDX=1",
        "CANCERDX=2",
        "DIABDX=-1",
        "DIABDX=1",
        "DIABDX=2",
        "JTPAIN=-1",
        "JTPAIN=1",
        "JTPAIN=2",
        "ARTHDX=-1",
        "ARTHDX=1",
        "ARTHDX=2",
        "ARTHTYPE=-1",
        "ARTHTYPE=1",
        "ARTHTYPE=2",
        "ARTHTYPE=3",
        "ASTHDX=1",
        "ASTHDX=2",
        "ADHDADDX=-1",
        "ADHDADDX=1",
        "ADHDADDX=2",
        "PREGNT=-1",
        "PREGNT=1",
        "PREGNT=2",
        "WLKLIM=-1",
        "WLKLIM=1",
        "WLKLIM=2",
        "ACTLIM=-1",
        "ACTLIM=1",
        "ACTLIM=2",
        "SOCLIM=-1",
        "SOCLIM=1",
        "SOCLIM=2",
        "COGLIM=-1",
        "COGLIM=1",
        "COGLIM=2",
        "DFHEAR42=-1",
        "DFHEAR42=1",
        "DFHEAR42=2",
        "DFSEE42=-1",
        "DFSEE42=1",
        "DFSEE42=2",
        "ADSMOK42=-1",
        "ADSMOK42=1",
        "ADSMOK42=2",
        "PHQ242=-1",
        "PHQ242=0",
        "PHQ242=1",
        "PHQ242=2",
        "PHQ242=3",
        "PHQ242=4",
        "PHQ242=5",
        "PHQ242=6",
        "EMPST=-1",
        "EMPST=1",
        "EMPST=2",
        "EMPST=3",
        "EMPST=4",
        "POVCAT=1",
        "POVCAT=2",
        "POVCAT=3",
        "POVCAT=4",
        "POVCAT=5",
        "INSCOV=1",
        "INSCOV=2",
        "INSCOV=3",
        "RACE",
    ]

    y = df[response_name].values.astype(np.float32)
    X = df[col_names].values.astype(np.float32)

    return X, y


def load_meps20():
    df = pd.read_csv(BASE_PATH + "meps_20_reg.csv")
    column_names = df.columns
    response_name = "UTILIZATION_reg"
    column_names = column_names[column_names != response_name]
    column_names = column_names[column_names != "Unnamed: 0"]

    col_names = [
        "AGE",
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PERWT15F",
        "REGION=1",
        "REGION=2",
        "REGION=3",
        "REGION=4",
        "SEX=1",
        "SEX=2",
        "MARRY=1",
        "MARRY=2",
        "MARRY=3",
        "MARRY=4",
        "MARRY=5",
        "MARRY=6",
        "MARRY=7",
        "MARRY=8",
        "MARRY=9",
        "MARRY=10",
        "FTSTU=-1",
        "FTSTU=1",
        "FTSTU=2",
        "FTSTU=3",
        "ACTDTY=1",
        "ACTDTY=2",
        "ACTDTY=3",
        "ACTDTY=4",
        "HONRDC=1",
        "HONRDC=2",
        "HONRDC=3",
        "HONRDC=4",
        "RTHLTH=-1",
        "RTHLTH=1",
        "RTHLTH=2",
        "RTHLTH=3",
        "RTHLTH=4",
        "RTHLTH=5",
        "MNHLTH=-1",
        "MNHLTH=1",
        "MNHLTH=2",
        "MNHLTH=3",
        "MNHLTH=4",
        "MNHLTH=5",
        "HIBPDX=-1",
        "HIBPDX=1",
        "HIBPDX=2",
        "CHDDX=-1",
        "CHDDX=1",
        "CHDDX=2",
        "ANGIDX=-1",
        "ANGIDX=1",
        "ANGIDX=2",
        "MIDX=-1",
        "MIDX=1",
        "MIDX=2",
        "OHRTDX=-1",
        "OHRTDX=1",
        "OHRTDX=2",
        "STRKDX=-1",
        "STRKDX=1",
        "STRKDX=2",
        "EMPHDX=-1",
        "EMPHDX=1",
        "EMPHDX=2",
        "CHBRON=-1",
        "CHBRON=1",
        "CHBRON=2",
        "CHOLDX=-1",
        "CHOLDX=1",
        "CHOLDX=2",
        "CANCERDX=-1",
        "CANCERDX=1",
        "CANCERDX=2",
        "DIABDX=-1",
        "DIABDX=1",
        "DIABDX=2",
        "JTPAIN=-1",
        "JTPAIN=1",
        "JTPAIN=2",
        "ARTHDX=-1",
        "ARTHDX=1",
        "ARTHDX=2",
        "ARTHTYPE=-1",
        "ARTHTYPE=1",
        "ARTHTYPE=2",
        "ARTHTYPE=3",
        "ASTHDX=1",
        "ASTHDX=2",
        "ADHDADDX=-1",
        "ADHDADDX=1",
        "ADHDADDX=2",
        "PREGNT=-1",
        "PREGNT=1",
        "PREGNT=2",
        "WLKLIM=-1",
        "WLKLIM=1",
        "WLKLIM=2",
        "ACTLIM=-1",
        "ACTLIM=1",
        "ACTLIM=2",
        "SOCLIM=-1",
        "SOCLIM=1",
        "SOCLIM=2",
        "COGLIM=-1",
        "COGLIM=1",
        "COGLIM=2",
        "DFHEAR42=-1",
        "DFHEAR42=1",
        "DFHEAR42=2",
        "DFSEE42=-1",
        "DFSEE42=1",
        "DFSEE42=2",
        "ADSMOK42=-1",
        "ADSMOK42=1",
        "ADSMOK42=2",
        "PHQ242=-1",
        "PHQ242=0",
        "PHQ242=1",
        "PHQ242=2",
        "PHQ242=3",
        "PHQ242=4",
        "PHQ242=5",
        "PHQ242=6",
        "EMPST=-1",
        "EMPST=1",
        "EMPST=2",
        "EMPST=3",
        "EMPST=4",
        "POVCAT=1",
        "POVCAT=2",
        "POVCAT=3",
        "POVCAT=4",
        "POVCAT=5",
        "INSCOV=1",
        "INSCOV=2",
        "INSCOV=3",
        "RACE",
    ]

    y = df[response_name].values.astype(np.float32)
    X = df[col_names].values.astype(np.float32)
    return X, y


def load_meps21():
    df = pd.read_csv(BASE_PATH + "meps_21_reg.csv")
    column_names = df.columns
    response_name = "UTILIZATION_reg"
    column_names = column_names[column_names != response_name]
    column_names = column_names[column_names != "Unnamed: 0"]

    col_names = [
        "AGE",
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PERWT16F",
        "REGION=1",
        "REGION=2",
        "REGION=3",
        "REGION=4",
        "SEX=1",
        "SEX=2",
        "MARRY=1",
        "MARRY=2",
        "MARRY=3",
        "MARRY=4",
        "MARRY=5",
        "MARRY=6",
        "MARRY=7",
        "MARRY=8",
        "MARRY=9",
        "MARRY=10",
        "FTSTU=-1",
        "FTSTU=1",
        "FTSTU=2",
        "FTSTU=3",
        "ACTDTY=1",
        "ACTDTY=2",
        "ACTDTY=3",
        "ACTDTY=4",
        "HONRDC=1",
        "HONRDC=2",
        "HONRDC=3",
        "HONRDC=4",
        "RTHLTH=-1",
        "RTHLTH=1",
        "RTHLTH=2",
        "RTHLTH=3",
        "RTHLTH=4",
        "RTHLTH=5",
        "MNHLTH=-1",
        "MNHLTH=1",
        "MNHLTH=2",
        "MNHLTH=3",
        "MNHLTH=4",
        "MNHLTH=5",
        "HIBPDX=-1",
        "HIBPDX=1",
        "HIBPDX=2",
        "CHDDX=-1",
        "CHDDX=1",
        "CHDDX=2",
        "ANGIDX=-1",
        "ANGIDX=1",
        "ANGIDX=2",
        "MIDX=-1",
        "MIDX=1",
        "MIDX=2",
        "OHRTDX=-1",
        "OHRTDX=1",
        "OHRTDX=2",
        "STRKDX=-1",
        "STRKDX=1",
        "STRKDX=2",
        "EMPHDX=-1",
        "EMPHDX=1",
        "EMPHDX=2",
        "CHBRON=-1",
        "CHBRON=1",
        "CHBRON=2",
        "CHOLDX=-1",
        "CHOLDX=1",
        "CHOLDX=2",
        "CANCERDX=-1",
        "CANCERDX=1",
        "CANCERDX=2",
        "DIABDX=-1",
        "DIABDX=1",
        "DIABDX=2",
        "JTPAIN=-1",
        "JTPAIN=1",
        "JTPAIN=2",
        "ARTHDX=-1",
        "ARTHDX=1",
        "ARTHDX=2",
        "ARTHTYPE=-1",
        "ARTHTYPE=1",
        "ARTHTYPE=2",
        "ARTHTYPE=3",
        "ASTHDX=1",
        "ASTHDX=2",
        "ADHDADDX=-1",
        "ADHDADDX=1",
        "ADHDADDX=2",
        "PREGNT=-1",
        "PREGNT=1",
        "PREGNT=2",
        "WLKLIM=-1",
        "WLKLIM=1",
        "WLKLIM=2",
        "ACTLIM=-1",
        "ACTLIM=1",
        "ACTLIM=2",
        "SOCLIM=-1",
        "SOCLIM=1",
        "SOCLIM=2",
        "COGLIM=-1",
        "COGLIM=1",
        "COGLIM=2",
        "DFHEAR42=-1",
        "DFHEAR42=1",
        "DFHEAR42=2",
        "DFSEE42=-1",
        "DFSEE42=1",
        "DFSEE42=2",
        "ADSMOK42=-1",
        "ADSMOK42=1",
        "ADSMOK42=2",
        "PHQ242=-1",
        "PHQ242=0",
        "PHQ242=1",
        "PHQ242=2",
        "PHQ242=3",
        "PHQ242=4",
        "PHQ242=5",
        "PHQ242=6",
        "EMPST=-1",
        "EMPST=1",
        "EMPST=2",
        "EMPST=3",
        "EMPST=4",
        "POVCAT=1",
        "POVCAT=2",
        "POVCAT=3",
        "POVCAT=4",
        "POVCAT=5",
        "INSCOV=1",
        "INSCOV=2",
        "INSCOV=3",
        "RACE",
    ]

    y = df[response_name].values.astype(np.float32)
    X = df[col_names].values.astype(np.float32)

    return X, y


def load_blog_data():
    # https://github.com/xinbinhuang/feature-selection_blogfeedback
    df = pd.read_csv(BASE_PATH + "blogData_train.csv", header=None)
    X = df.iloc[:, 0:280].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    return X, y


def load_bike():
    # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
    df = pd.read_csv(BASE_PATH + "bike_train.csv")

    # # seperating season as per values. this is bcoz this will enhance features.
    season = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season], axis=1)

    # # # same for weather. this is bcoz this will enhance features.
    weather = pd.get_dummies(df["weather"], prefix="weather")
    df = pd.concat([df, weather], axis=1)

    # # # now can drop weather and season.
    df.drop(["season", "weather"], inplace=True, axis=1)
    df.head()

    df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
    df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
    df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
    df["year"] = [t.year for t in pd.DatetimeIndex(df.datetime)]
    df["year"] = df["year"].map({2011: 0, 2012: 1})

    df.drop("datetime", axis=1, inplace=True)
    df.drop(["casual", "registered"], axis=1, inplace=True)
    df.columns.to_series().groupby(df.dtypes).groups
    X = df.drop("count", axis=1).values.astype(np.float32)
    y = df["count"].values.astype(np.float32)

    return X, y


def load_community():
    # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
    attrib = pd.read_csv(
        BASE_PATH + "communities_attributes.csv", delim_whitespace=True
    )
    data = pd.read_csv(BASE_PATH + "communities.data", names=attrib["attributes"])
    data = data.drop(
        columns=["state", "county", "community", "communityname", "fold"], axis=1
    )

    data = data.replace("?", np.nan)

    # Impute mean values for samples with missing values
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    imputer = imputer.fit(data[["OtherPerCap"]])
    data[["OtherPerCap"]] = imputer.transform(data[["OtherPerCap"]])
    data = data.dropna(axis=1)
    X = data.iloc[:, 0:100].values.astype(np.float32)
    y = data.iloc[:, 100].values.astype(np.float32)
    return X, y


def load_facebook1():
    df = pd.read_csv(BASE_PATH + "facebook/Features_Variant_1.csv")
    y = df.iloc[:, 53].values.astype(np.float32)
    X = df.iloc[:, 0:53].values.astype(np.float32)
    return X, y


def load_facebook2():
    df = pd.read_csv(BASE_PATH + "facebook/Features_Variant_2.csv")
    y = df.iloc[:, 53].values.astype(np.float32)
    X = df.iloc[:, 0:53].values.astype(np.float32)
    return X, y


def _cond_exp(x):
    return np.sin(1 / (x[:, 0] ** 3))


def _noise_sd_fn(x):
    return 1 * x[:, 0] ** 2


def _generate_data(
    n, p, cond_exp, noise_sd_fn, x_dist=partial(np.random.uniform, low=0, high=10)
):
    """
    generate data from a conditional expectation function and a noise function
    :param n: number of data points
    :param p: number of features
    :param cond_exp: conditional expectation function
    :param noise_sd_fn: noise standard deviation function
    :param x_dist: distribution of x
    :return: generated data
    """
    x = x_dist(size=n * p).reshape(n, p).astype(np.float32)
    noise_sd = noise_sd_fn(x)
    noise = np.random.normal(scale=noise_sd, size=n)
    y = (cond_exp(x) + noise).reshape(-1, 1).astype(np.float32)
    return x, y


def load_wave_data(n_train, n_test, n_calib):
    """
    Load the wave toy dataset
    :param n_train: number of training data points
    :param n_test: number of tests data points
    :param n_calib: number of calibration data points
    :return: data
    """
    p = 1
    x_dist = partial(np.random.beta, a=1.2, b=0.8)
    X, y = _generate_data(
        n_train + n_test + n_calib, p, _cond_exp, _noise_sd_fn, x_dist=x_dist
    )

    x_train, y_train = X[:n_train], y[:n_train]
    x_calib, y_calib = X[n_train : n_train + n_calib], y[n_train : n_train + n_calib]
    x_test, y_test = X[n_train + n_calib :], y[n_train + n_calib :]

    # Normalize
    x_train, norm_params = min_max_normalize(x_train)
    x_calib = (x_calib - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
    x_test = (x_test - norm_params["min"]) / (norm_params["max"] - norm_params["min"])

    if n_calib == 0:
        return x_train, y_train, x_test, y_test
    return x_train, y_train, x_test, y_test, x_calib, y_calib


def load_heteroscedastic_data(n_train, n_test):
    def f(x):
        """Construct data (1D example)"""
        ax = 0 * x
        for i in range(len(x)):
            ax[i] = np.random.poisson(np.sin(x[i]) ** 2 + 0.1) + 0.03 * x[
                i
            ] * np.random.randn(1)
            ax[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return ax.astype(np.float32)

    # training features
    x_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)

    # tests features
    x_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)

    # generate labels
    y_train = f(x_train).reshape(-1, 1)
    y_test = f(x_test).reshape(-1, 1)

    # reshape the features
    x_train = np.reshape(x_train, (n_train, 1))
    x_test = np.reshape(x_test, (n_test, 1))

    # Normalize
    x_train, norm_params = min_max_normalize(x_train)
    x_test = (x_test - norm_params["min"]) / (norm_params["max"] - norm_params["min"])

    return x_train, y_train, x_test, y_test


###########################################################
# Load Regression toy data                                #
# From:https://github.com/aamini/evidential-deep-learning #
###########################################################

data_dir = ROOT_PATH + "uci/"


def _load_boston():
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    data = np.loadtxt(os.path.join(data_dir, "boston-housing/boston_housing.txt"))
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def _load_powerplant():
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(data_dir, "power-plant/Folds5x2_pp.xlsx")
    data = pd.read_excel(data_file)
    x = data.values[:, :-1]
    y = data.values[:, -1]
    return x, y


def _load_concrete():
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, "concrete/Concrete_Data.xls")
    data = pd.read_excel(data_file)
    X = data.values[:, :-1].astype(np.float32)
    y = data.values[:, -1].astype(np.float32)
    return X, y


def _load_yacht():
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(data_dir, "yacht/yacht_hydrodynamics.data")
    data = pd.read_csv(data_file, delim_whitespace=True)
    X = data.values[:, :-1]
    y = data.values[:, -1]
    return X, y


def _load_energy_efficiency():
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(data_dir, "energy-efficiency/ENB2012_data.xlsx")
    data = pd.read_excel(data_file)
    X = data.values[:, :-2]
    # Optionally, you can load y_heating as well
    # y_heating = data.values[:, -2]
    y_cooling = data.values[:, -1]
    return X, y_cooling


def _load_wine():
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    # data_file = os.path.join(data_dir, 'wine-quality/winequality-red.csv')
    data_file = os.path.join(data_dir, "wine-quality/wine_data_new.txt")
    data = pd.read_csv(data_file, sep=" ", header=None)
    X = data.values[:, :-1]
    y = data.values[:, -1]
    return X, y


def _load_kin8nm():
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.

    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).

    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(data_dir, "kin8nm/dataset_2175_kin8nm.csv")
    data = pd.read_csv(data_file, sep=",")
    X = data.values[:, :-1]
    y = data.values[:, -1]
    return X, y


def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = np.loadtxt(os.path.join(data_dir, "naval/data.txt"))
    X = data[:, :-2]
    # Optionally, you can load y_compressor as well
    # y_compressor = data[:, -2]
    y_turbine = data[:, -1]
    return X, y_turbine


def _load_protein():
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

    TODO: Check that the output is correct

    Input variables:
        RMSD-Size of the residue.
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
    Output variable:
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(data_dir, "protein/CASP.csv")
    data = pd.read_csv(data_file, sep=",")
    X = data.values[:, 1:]
    y = data.values[:, 0]
    return X, y


def _load_song():
    """
    INSTRUCTIONS:
    1) Download from http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    2) Place YearPredictionMSD.txt in data/uci/song/

    Dataloader is slow since file is large.

    YearPredictionMSD Data Set
    Prediction of the release year of a song from audio features. Songs are mostly western, commercial tracks ranging
    from 1922 to 2011, with a peak in the year 2000s.

    90 attributes, 12 = timbre average, 78 = timbre covariance
    The first value is the year (target), ranging from 1922 to 2011.
    Features extracted from the 'timbre' features from The Echo Nest API.
    We take the average and covariance over all 'segments', each segment
    being described by a 12-dimensional timbre vector.

    """
    data = np.loadtxt(
        os.path.join(data_dir, "song/YearPredictionMSD.txt"), delimiter=","
    )
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def _load_depth():
    train = h5py.File("data/depth_train.h5", "r")
    test = h5py.File("data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])


def load_depth():
    return _load_depth()


def load_apollo():
    test = h5py.File("../data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])


def load_dataset(name, split_seed=0, test_fraction=0.1, return_as_tensor=False):
    # load full dataset
    load_funs = {
        "wine": _load_wine,
        "boston": _load_boston,
        "concrete": _load_concrete,
        "power-plant": _load_powerplant,
        "yacht": _load_yacht,
        "energy-efficiency": _load_energy_efficiency,
        "kin8nm": _load_kin8nm,
        "naval": _load_naval,
        "protein": _load_protein,
        "depth": _load_depth,
        "song": _load_song,
    }

    print("Loading dataset {}....".format(name))
    if name == "depth":
        (X_train, y_train), (X_test, y_test) = load_funs[name]()
        y_scale = np.array([[1.0]])
        return (X_train, y_train), (X_test, y_test), y_scale

    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    # We create the train and tests sets with 90% and 10% of the data

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    if name == "boston" or name == "wine":
        test_fraction = 0.2
    size_train = int(np.round(X.shape[0] * (1 - test_fraction)))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    X_train = X[index_train, :]
    X_test = X[index_test, :]

    if name == "depth":
        y_train = y[index_train]
        y_test = y[index_test]
    else:
        y_train = y[index_train, None]
        y_test = y[index_test, None]

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale

    if return_as_tensor:
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

    print("Done loading dataset {}".format(name))
    return (X_train, y_train), (X_test, y_test), y_train_scale


def load_flight_delay():

    # Download from here: http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/dataset_mirror/airline_delay/
    data = pd.read_pickle("../data/flight-delay/filtered_data.pickle")
    y = np.array(data["ArrDelay"])
    data.pop("ArrDelay")
    X = np.array(data[:])

    def standardize(data):
        data -= data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        data /= scale
        return data, scale

    X = X[:, np.where(data.var(axis=0) > 0)[0]]
    X, _ = standardize(X)
    y, y_scale = standardize(y.reshape(-1, 1))
    y = np.squeeze(y)
    # y_scale = np.array([[1.0]])

    N = 700000
    S = 100000
    X_train = X[:N, :]
    X_test = X[N : N + S, :]
    y_train = y[:N]
    y_test = y[N : N + S]

    return (X_train, y_train), (X_test, y_test), y_scale

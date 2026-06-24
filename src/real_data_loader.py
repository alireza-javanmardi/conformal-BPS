"""
The purpose of this file is to include all relevant code for creating the test_loader for cifar10, chaosnli, and qualitymri datasets. 
This file is adopted from https://github.com/timoverse/credal-prediction-relative-likelihood/blob/main/data.py
"""


import sys
import os
import torch
import pickle
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.model_selection import train_test_split
from probly.data import DCICDataset
from PIL import Image
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)
data_path = os.path.join(root_path, "data") 

def get_test_loader(name, seed, batch_size=128, **kwargs):

    rng = torch.Generator().manual_seed(seed)

    # --------------------------------------------------
    # 1) LOAD RAW DATA
    # --------------------------------------------------

    if name == "cifar10":

        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

        soft_ds = np.load(os.path.join(data_path, "CIFAR-10-H", "cifar10h-probs.npy"))
        hard_ds = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms)

        def build_item(idx):
            p = soft_ds[idx]
            x, y = hard_ds[idx]

            return {
                "x": x,
                "y": torch.tensor(y, dtype=torch.long),
                "p": torch.tensor(p, dtype=torch.float32)
            }

        N = len(soft_ds)

    # --------------------------------------------------
    elif name == "cifar10c":
        corruption = kwargs.get("corruption", "gaussian_noise")
        severity = kwargs.get("severity", 1)

        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

        images_np = np.load(os.path.join(data_path, "CIFAR-10-C", f"{corruption}.npy"))
        labels_np = np.load(os.path.join(data_path, "CIFAR-10-C", "labels.npy"))
        soft_ds   = np.load(os.path.join(data_path, "CIFAR-10-H", "cifar10h-probs.npy"))

        start = (severity - 1) * 10_000
        end   = severity * 10_000
        images_np = images_np[start:end]
        labels_np = labels_np[start:end]
        # soft_ds is NOT sliced — it always covers the 10 000 clean test images,
        # indexed the same way as CIFAR-10-C's labels.npy

        def build_item(idx):
            img = Image.fromarray(images_np[idx])
            x = transforms(img)
            y = int(labels_np[idx])
            p = soft_ds[idx]

            return {
                "x": x,
                "y": torch.tensor(y, dtype=torch.long),
                "p": torch.tensor(p, dtype=torch.float32),
            }

        N = len(labels_np)
    # --------------------------------------------------

    elif name == "chaosnli":

        with open(os.path.join(data_path, "chaosnli", "embeddings", "snli.pkl"), "rb") as f:
            snli = pickle.load(f)
        with open(os.path.join(data_path, "chaosnli", "embeddings", "mnli_m.pkl"), "rb") as f:
            mnli = pickle.load(f)

        embedding = np.concatenate((snli["embedding"], mnli["embedding"]), axis=0)
        label_dist = np.concatenate((snli["label_dist"], mnli["label_dist"]), axis=0)
        premise = np.concatenate((snli["premise"], mnli["premise"]), axis=0)
        hypothesis = np.concatenate((snli["hypothesis"], mnli["hypothesis"]), axis=0)

        _, X_test, _, p_test, _, premise_test, _, hypothesis_test = train_test_split(embedding, label_dist, premise, hypothesis, test_size=0.2, random_state=seed)

        def build_item(idx):
            x = torch.tensor(X_test[idx], dtype=torch.float32)
            p = torch.tensor(p_test[idx], dtype=torch.float32)
            y = torch.multinomial(p, 1).item()

            return {
                "x": x,
                "y": torch.tensor(y, dtype=torch.long),
                "p": p,
                "premise-hypothesis": (premise_test[idx], hypothesis_test[idx])
            }

        N = len(X_test)

    # --------------------------------------------------

    elif name == "qualitymri":

        transforms = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize((0.1485, 0.1485, 0.1485),
                        (0.1819, 0.1819, 0.1819))
        ])

        base = DCICDataset(
            root=os.path.join(data_path, "QualityMRI"),
            transform=transforms,
            first_order=True
        )

        _, test_subset = torch.utils.data.random_split(base, [0.8, 0.2], generator=rng)

        def build_item(idx):
            x, p = test_subset[idx]
            p = torch.tensor(p, dtype=torch.float32)
            y = torch.multinomial(p, 1).item()

            return {
                "x": x,
                "y": torch.tensor(y, dtype=torch.long),
                "p": p
            }

        N = len(test_subset)

    else:
        raise ValueError(f"{name} not supported")

    # --------------------------------------------------
    # 2) GENERIC DATASET WRAPPER
    # --------------------------------------------------

    class GenericDataset(torch.utils.data.Dataset):
        def __len__(self):
            return N

        def __getitem__(self, idx):
            return build_item(idx)

    dataset = GenericDataset()

    # --------------------------------------------------
    # 3) LOADER
    # --------------------------------------------------

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
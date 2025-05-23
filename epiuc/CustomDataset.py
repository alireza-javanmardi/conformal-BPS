from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        n_samples1, *n_features = X.shape
        n_samples2 = len(Y)

        if n_samples1 != n_samples2:
            raise ValueError("X and Y must have same amount of samples")

        self.n_samples = n_samples1
        self.n_features = n_features

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        if self.transform:
            return self.transform(self.X[idx]), self.Y[idx]
        return self.X[idx], self.Y[idx]

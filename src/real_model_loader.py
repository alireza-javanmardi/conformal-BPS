"""
The purpose of this file is to include all relevant code for loading credal set predictors. 
This file is adopted from https://github.com/timoverse/credal-prediction-relative-likelihood/blob/main/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from probly.representation import Ensemble
from tqdm import tqdm


def get_model(base, n_classes):
    if base == 'resnet':
        model = ResNet18()
        model.linear = nn.Linear(512, n_classes)
    elif base == 'fcnet':
        model = FCNet(768, n_classes)
    elif base == 'torchresnet':
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(f"Unknown base model: {base}")
    return model

class LikelihoodEnsemble(Ensemble):
    def __init__(self, base, n_classes, n_members, tobias_value=100):
        super().__init__(base, n_members)
        self.n_members = n_members
        self.rls = [1.0]
        self.tobias_value = tobias_value
        if self.tobias_value:
            tobias_init_ensemble(self, n_classes, tobias_value)


class DesterckeEnsemble(Ensemble):
    def __init__(self, base, n_members):
        super().__init__(base, n_members)
        self.n_members = n_members

    @torch.no_grad()
    def predict_representation(self, x: torch.Tensor, alpha: float, distance: str, logits: bool = False) -> torch.Tensor:
        x = super().predict_representation(x, logits)
        if distance == 'euclidean':
            # when the distance is euclidean the mean is the representative probability distribution
            representative = torch.mean(x, dim=1)
            # compute distances to the representative distribution
            dists = torch.cdist(x, torch.unsqueeze(representative, 1), p=2)
            # discard alpha percent of the predictions with the largest distances
            # sort the distances
            sorted_indices = torch.argsort(dists.squeeze(), dim=1)
            # get the indices of the predictions to keep
            keep_indices = sorted_indices[:, :int(round((1 - alpha) * self.n_members))]
            # get the predictions to keep
            keep_predictions = torch.gather(x, 1, keep_indices.unsqueeze(2).expand(-1, -1, x.shape[2]))
        else:
            raise ValueError(f"Unknown distance metric: {distance}")
        return keep_predictions


class FCNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def tobias_init_ensemble(ensemble, n_classes, value=100):
    for i in range(1, len(ensemble.models)):
        tobias_initialization(ensemble.models[i], (i-1) % n_classes, value)

def tobias_initialization(model, clss, value=100):
    last_layer = list(module for module in model.modules() if isinstance(module, nn.Linear))[-1]
    last_layer.bias.data[clss] = value

def load_ensemble(ensemble, path):
    for i in range(len(ensemble.models)):
        dict_path = f'{path}_state_dict_{i}.pt'
        ensemble.models[i].load_state_dict(torch.load(dict_path))
    rl_path = f'{path}_rls.pt'
    ensemble.rls = torch.load(rl_path)


@torch.no_grad()
def torch_get_outputs(model, loader, device):
    inputs = torch.empty(0, device=device)
    targets_y = torch.empty(0, device=device)
    targets_p = torch.empty(0, device=device)
    outputs = torch.empty(0, device=device)
    for batch in tqdm(loader):
        input, target_y, target_p = batch["x"].to(device), batch["y"].to(device), batch["p"].to(device)
        inputs = torch.cat((inputs, input), dim=0)
        targets_y = torch.cat((targets_y, target_y), dim=0)
        targets_p = torch.cat((targets_p, target_p), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input)), dim=0)
    return inputs, targets_y, targets_p, outputs    

@torch.no_grad()
def torch_get_outputs_destercke(model, loader, device, alpha):
    inputs = torch.empty(0, device=device)
    targets_y = torch.empty(0, device=device)
    targets_p = torch.empty(0, device=device)
    outputs = torch.empty(0, device=device)
    for batch in tqdm(loader):
        input, target_y, target_p = batch["x"].to(device), batch["y"].to(device), batch["p"].to(device)
        inputs = torch.cat((inputs, input), dim=0)
        targets_y = torch.cat((targets_y, target_y), dim=0)
        targets_p = torch.cat((targets_p, target_p), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input, alpha, 'euclidean')), dim=0)
    return inputs, targets_y, targets_p, outputs
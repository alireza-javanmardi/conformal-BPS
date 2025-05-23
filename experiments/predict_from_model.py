import sys
import os
import torch
import numpy as np
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from epiuc.utils.data_load import load_cifar100, load_cifar10

from epiuc.uncertainty.classification import Resnet50_Cifar10, Resnet50_Cifar100, Resnet50_Imagenet
from epiuc.uncertainty.wrapper import Evidential_Classifier, Ensemble_Classifier, MC_Classifier
from epiuc.config import BACKEND
from torch.distributions import Dirichlet


data = sys.argv[1]
model = sys.argv[2]

#############################
#Load data and base model
#############################
conf = {
    "batch_size": 128,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": True,
}

if data == "cifar10": 
    _ , _ , testloader = load_cifar10(vali_size=0, loading_configs= conf)
    resnet_50 = Resnet50_Cifar10(random_state=42)
elif data == "cifar100":
    _ , _ , testloader = load_cifar100(vali_size=0, loading_configs= conf)
    resnet_50 = Resnet50_Cifar100(random_state=42)
elif data == "imagenet": 
    val_dir = os.path.join(root_path, "data", "Imagenet", "imagenet_validation")  # wherever your split folders are
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    testloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    resnet_50 = Resnet50_Imagenet(random_state=42)

images  =[]
labels = []
for (img, label) in testloader:
    images.append(img)
    labels.append(label)

images = torch.cat(images, dim=0)
labels = torch.cat(labels, dim=0)

#############################
#Get predicted distributions
#############################
if model == "ensemble": 
    ua_model = Ensemble_Classifier(base_model=resnet_50, n_models=5, random_state=42)
    ua_model.load(os.path.join(root_path, "models", data.upper(), "Ensemble_Classification/"))
    ua_model.compile(backend=BACKEND)
    ua_model.set_device(torch.device("cuda")) #comment if using cpu
    probs, TAEUs = ua_model.predict(testloader, raw_output=True)
    

elif model == "mc":
    ua_model = MC_Classifier(base_model=resnet_50, random_state=42, n_iterations=5, dropout_prob=0.05,)
    ua_model.load(os.path.join(root_path, "models", data.upper(), "MC_Classification_on_Resnet50_"+data.capitalize()+".pth"))
    ua_model.compile(backend=BACKEND)
    ua_model.set_device(torch.device("cuda")) #comment if using cpu
    probs, TAEUs = ua_model.predict(testloader, raw_output=True)

elif model == "evidential":
    ua_model = Evidential_Classifier(base_model=resnet_50, evidence_method="softplus", random_state=42)
    ua_model.load(os.path.join(root_path, "models", data.upper(), "Evidential_Classifier_softplus_with_Resnet50_"+data.capitalize()+".pth"))
    ua_model.compile(backend=BACKEND)
    ua_model.set_device(torch.device("cuda")) #comment if using cpu
    evi_alphas, TAEUs = ua_model.predict(testloader, raw_output=True)
    TAEUs = TAEUs.squeeze()
    probs = torch.stack([Dirichlet(dir_alpha).sample((5,)) for dir_alpha in evi_alphas])

mean_probs = probs.mean(axis=1).unsqueeze(1)

#############################
#save predictions
#############################
predictions = {
    "labels": labels,
    "probs": probs,
    "mean_probs": mean_probs,
    "TAEUs" : TAEUs
}
if model == "evidential":
    predictions["evi_alphas"] = evi_alphas

os.makedirs(os.path.join(root_path, "all_results", data, model), exist_ok=True) 
# with open(os.path.join(root_path, "all_results", data, model, "predictions.pkl"), 'wb') as f:
#     pickle.dump(predictions, f)
with open(os.path.join(root_path, "all_results", data, model, "predictions.pt"), 'wb') as f:
    torch.save(predictions, f)
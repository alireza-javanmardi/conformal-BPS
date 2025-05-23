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

from epiuc.conformal_prediction.uacp import avg_true_label_inclusion
from epiuc.conformal_prediction.eval_metrics import covGap_nonbinary, size_stratified_cov_violation_nonbinary
from torch.distributions import Dirichlet
from sklearn.model_selection import train_test_split


model = sys.argv[1]
alpha_str = sys.argv[2]
alpha = float(alpha_str)
alpha_credal_str = sys.argv[3]
alpha_credal = float(alpha_credal_str)
calib_size_str = sys.argv[4]
calib_size = float(calib_size_str)


data ="cifar10"
#############################
#Load predictions
#############################
with open(os.path.join(root_path, "all_results", data, model, "predictions.pkl"), 'rb') as f:
    predictions = pickle.load(f)
labels = predictions["labels"]
probs = predictions["probs"]
mean_probs = predictions["mean_probs"]

#############################
#Get results and save them
#############################
annotations_prob = np.load(os.path.join(root_path, "data", "CIFAR-10-H", "cifar10h-probs.npy"))
results = {}
for r in ["marginal cvg", "set size", "conditional cvg", "coverage gap", "SSCV"]:
    results[r] = {"BPS": [], "APS": [], "BPS_cons": [], "APS_cons": [], "BPS_nominal": [], "APS_nominal": []}

for exp_seed in range(10):
    exp_seed_str = str(exp_seed)
    calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels, calib_annotations, test_annotations = train_test_split(probs.numpy(), mean_probs.numpy(), labels.numpy(), annotations_prob, train_size=calib_size, random_state=exp_seed)
    with open(os.path.join(root_path, "all_results", data, "credal", model, "calib_size "+calib_size_str, "alpha credal "+alpha_credal_str,"alpha "+alpha_str, "seed " + exp_seed_str, "res.pkl"), 'rb') as f:
        res = pickle.load(f)
    for approach in ["BPS", "APS", "BPS_cons", "APS_cons", "BPS_nominal", "APS_nominal"]:
        results["marginal cvg"][approach].append(avg_true_label_inclusion(res["opt_b"][approach], test_labels))
        results["set size"][approach].append(np.mean(np.sum(res["opt_b"][approach], axis=1)))
        results["conditional cvg"][approach].append(np.mean(np.sum(np.multiply(res["opt_b"][approach], test_annotations), axis=1)))
        results["coverage gap"][approach].append(covGap_nonbinary(res["opt_b"][approach], test_labels, alpha, num_classes=mean_probs.shape[-1]))  
        results["SSCV"][approach].append(size_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, alpha, stratified_size= [[i, i+1] for i in range(mean_probs.shape[-1])]))  

with open(os.path.join(root_path, "all_results", data, "credal", model, "calib_size "+calib_size_str, "alpha credal "+alpha_credal_str, "alpha "+alpha_str, "results.pkl"), 'wb') as f:
    pickle.dump(results, f)

import sys
import os
import torch
import numpy as np
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from epiuc.conformal_prediction.uacp import avg_true_label_inclusion
from epiuc.conformal_prediction.eval_metrics import covGap_nonbinary, size_stratified_cov_violation_nonbinary, uncertainty_stratified_cov_violation_nonbinary
from sklearn.model_selection import train_test_split



data = sys.argv[1]
model = sys.argv[2]
alpha_str = sys.argv[3]
alpha = float(alpha_str)
calib_size_str = sys.argv[4]
calib_size = float(calib_size_str)



#############################
#Load predictions
#############################

if data == "imagenet":
    with open(os.path.join(root_path, "all_results", data, model, "predictions.pt"), 'rb') as f:
        predictions = torch.load(f, map_location=torch.device('cpu'))
else: 
    with open(os.path.join(root_path, "all_results", data, model, "predictions.pkl"), 'rb') as f:
        predictions = pickle.load(f)
labels = predictions["labels"]
probs = predictions["probs"]
mean_probs = predictions["mean_probs"]
TAEUs = predictions["TAEUs"]
max_TAEUs = TAEUs.numpy().max(axis=0)
n = 50
#############################
#Get results
#############################
results = {}
if data == "cifar10": 
    annotations_prob = np.load(os.path.join(root_path, "data", "CIFAR-10-H", "cifar10h-probs.npy"))
    for r in ["marginal cvg", "set size", "conditional cvg", "coverage gap", "SSCV", "TUSCV", "EUSCV", "AUSCV"]:
        results[r] = {"BPS": [], "APS": [], "BPS_cons": [], "APS_cons": [], "BPS_nom": [], "APS_nom": []}

    for exp_seed in range(10):
        exp_seed_str = str(exp_seed)
        calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels, calib_annotations, test_annotations, calib_TAEUs, test_TAEUs= train_test_split(probs.numpy(), mean_probs.numpy(), labels.numpy(), annotations_prob, TAEUs.numpy(), train_size=calib_size, random_state=exp_seed)
        with open(os.path.join(root_path, "all_results", data, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "seed " + exp_seed_str, "res.pkl"), 'rb') as f:
            res = pickle.load(f)
        for approach in ["BPS", "APS", "BPS_cons", "APS_cons", "BPS_nom", "APS_nom"]:
            results["marginal cvg"][approach].append(avg_true_label_inclusion(res["opt_b"][approach], test_labels))
            results["set size"][approach].append(np.mean(np.sum(res["opt_b"][approach], axis=1)))
            results["conditional cvg"][approach].append(np.mean(np.sum(np.multiply(res["opt_b"][approach], test_annotations), axis=1)))
            results["coverage gap"][approach].append(covGap_nonbinary(res["opt_b"][approach], test_labels, alpha, num_classes=mean_probs.shape[-1]))  
            results["SSCV"][approach].append(size_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, alpha, stratified_size= [[i, i+1] for i in range(mean_probs.shape[-1])]))  
            results["TUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,0], alpha, stratified_size=[[i * max_TAEUs[0]/n, (i + 1) * max_TAEUs[0]/n] for i in range(n)]))
            results["EUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,1], alpha, stratified_size=[[i * max_TAEUs[1]/n, (i + 1) * max_TAEUs[1]/n] for i in range(n)]))
            results["AUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,2], alpha, stratified_size=[[i * max_TAEUs[2]/n, (i + 1) * max_TAEUs[2]/n] for i in range(n)]))
else: 
    for r in ["marginal cvg", "set size", "coverage gap", "SSCV", "TUSCV", "EUSCV", "AUSCV"]:
        results[r] = {"BPS": [], "APS": [], "BPS_cons": [], "APS_cons": [], "BPS_nom": [], "APS_nom": []}

    for exp_seed in range(10):
        exp_seed_str = str(exp_seed)
        # calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels = train_test_split(probs.numpy(), mean_probs.numpy(), labels.numpy(), train_size=calib_size, random_state=exp_seed)
        calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels, calib_TAEUs, test_TAEUs = train_test_split(probs.numpy(), mean_probs.numpy(), labels.numpy(), TAEUs.numpy(), train_size=calib_size, test_size=8000, random_state=exp_seed)
        with open(os.path.join(root_path, "all_results", data, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "seed " + exp_seed_str, "res.pkl"), 'rb') as f:
            res = pickle.load(f)
        for approach in ["BPS", "APS", "BPS_cons", "APS_cons", "BPS_nom", "APS_nom"]:
            results["marginal cvg"][approach].append(avg_true_label_inclusion(res["opt_b"][approach], test_labels))
            results["set size"][approach].append(np.mean(np.sum(res["opt_b"][approach], axis=1)))
            results["coverage gap"][approach].append(covGap_nonbinary(res["opt_b"][approach], test_labels, alpha, num_classes=mean_probs.shape[-1]))  
            results["SSCV"][approach].append(size_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, alpha, stratified_size= [[i, i+1] for i in range(mean_probs.shape[-1])]))  
            results["TUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,0], alpha, stratified_size=[[i * max_TAEUs[0]/n, (i + 1) * max_TAEUs[0]/n] for i in range(n)]))
            results["EUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,1], alpha, stratified_size=[[i * max_TAEUs[1]/n, (i + 1) * max_TAEUs[1]/n] for i in range(n)]))
            results["AUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(res["opt_b"][approach], test_labels, test_TAEUs[:,2], alpha, stratified_size=[[i * max_TAEUs[2]/n, (i + 1) * max_TAEUs[2]/n] for i in range(n)]))


with open(os.path.join(root_path, "all_results", data, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "results.pkl"), 'wb') as f:
    pickle.dump(results, f)

import sys
import os
import numpy as np
import pickle
import torch
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from epiuc.conformal_prediction.uacp import avg_true_label_inclusion
from epiuc.conformal_prediction.eval_metrics import covGap_nonbinary, size_stratified_cov_violation_nonbinary, uncertainty_stratified_cov_violation_nonbinary
from sklearn.model_selection import train_test_split
from epiuc.utils.uncertainty_decompositions import uncertainty_decomposition_entropy

data_size = sys.argv[1]
rarity_param = sys.argv[2]
data_seed_str = sys.argv[3]
data_seed = int(data_seed_str)
model = sys.argv[4]
alpha_str = sys.argv[5]
alpha = float(alpha_str)
calib_size_str = sys.argv[6]
calib_size = float(calib_size_str)




#############################
#Load data and base model
#############################
with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, "res.pkl"), 'rb') as f:
    res = pickle.load(f)

probs = res["probs"][model]
TAEUs = torch.stack(uncertainty_decomposition_entropy(probs), dim=1)
mean_probs = probs.mean(axis=1).unsqueeze(1)
labels = res["y_t"]
X_t = res["X_t"]
W_t = res["W_t"]
nominal_coverage = 1 - alpha
calibration_size = int(calib_size * len(labels))
CP_target = np.ceil(nominal_coverage*(calibration_size+1))/calibration_size

#############################
#load optimal sets and lambda
#############################
results = {}
for r in ["marginal cvg", "set size", "conditional cvg", "marginal cvg X1==1", "marginal cvg X1==-8", "set size X1==1", "set size X1==-8", "coverage gap", "SSCV", "TUSCV", "EUSCV", "AUSCV"]:
    results[r] = {"BPS": [], "APS": [], "BPS_cons": [], "APS_cons": [], "BPS_nom": [], "APS_nom": []}

for exp_seed in range(10):
    exp_seed_str = str(exp_seed)
    calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels, calib_X, test_X, calib_W, test_W, calib_TAEUs, test_TAEUs = train_test_split(probs.numpy(), mean_probs.numpy(), labels, X_t, W_t, TAEUs, train_size=calib_size, random_state=exp_seed)
    with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "seed " + exp_seed_str, "sets.pkl"), 'rb') as f:
        sets = pickle.load(f)
    for approach in ["BPS", "APS", "BPS_cons", "APS_cons", "BPS_nom", "APS_nom"]:
        results["marginal cvg"][approach].append(avg_true_label_inclusion(sets["opt_b"][approach], test_labels))
        results["marginal cvg X1==-8"][approach].append(avg_true_label_inclusion(sets["opt_b"][approach][test_X[:,0]==-8], test_labels[test_X[:,0]==-8]))
        results["marginal cvg X1==1"][approach].append(avg_true_label_inclusion(sets["opt_b"][approach][test_X[:,0]==1], test_labels[test_X[:,0]==1]))
        results["set size"][approach].append(np.mean(np.sum(sets["opt_b"][approach], axis=1)))
        results["set size X1==-8"][approach].append(np.mean(np.sum(sets["opt_b"][approach][test_X[:,0]==-8], axis=1)))
        results["set size X1==1"][approach].append(np.mean(np.sum(sets["opt_b"][approach][test_X[:,0]==1], axis=1)))
        results["conditional cvg"][approach].append(np.mean(np.sum(np.multiply(sets["opt_b"][approach], test_W), axis=1)))
        results["coverage gap"][approach].append(covGap_nonbinary(sets["opt_b"][approach], test_labels, alpha, num_classes=mean_probs.shape[-1]))  
        results["SSCV"][approach].append(size_stratified_cov_violation_nonbinary(sets["opt_b"][approach], test_labels, alpha, stratified_size= [[i, i+1] for i in range(mean_probs.shape[-1])]))  
        results["TUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(sets["opt_b"][approach], test_labels, test_TAEUs[:,0], alpha, stratified_size=[[i, i+0.01] for i in np.arange(0,1,0.01)]))
        results["EUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(sets["opt_b"][approach], test_labels, test_TAEUs[:,1], alpha, stratified_size=[[i, i+0.01] for i in np.arange(0,1,0.01)]))
        results["AUSCV"][approach].append(uncertainty_stratified_cov_violation_nonbinary(sets["opt_b"][approach], test_labels, test_TAEUs[:,2], alpha, stratified_size=[[i, i+0.01] for i in np.arange(0,1,0.01)]))
#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, model, "calib_size "+calib_size_str, "alpha "+alpha_str), exist_ok=True) 
with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "results.pkl"), 'wb') as f:
    pickle.dump(results, f)

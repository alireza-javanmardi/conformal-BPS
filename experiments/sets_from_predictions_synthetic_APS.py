import sys
import os
import torch
import numpy as np
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from epiuc.conformal_prediction.uacp import solve_b_in_batches, find_optimal_lambda
from sklearn.model_selection import train_test_split


data_size = sys.argv[1]
rarity_param = sys.argv[2]
data_seed_str = sys.argv[3]
data_seed = int(data_seed_str)
model = sys.argv[4]
alpha_str = sys.argv[5]
alpha = float(alpha_str)
calib_size_str = sys.argv[6]
calib_size = float(calib_size_str)
exp_seed_str = sys.argv[7]
exp_seed = int(exp_seed_str)



#############################
#Load data and base model
#############################
with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, "res.pkl"), 'rb') as f:
    res = pickle.load(f)


probs = res["probs"][model]
mean_probs = probs.mean(axis=1).unsqueeze(1)
labels = res["y_t"]
nominal_coverage = 1 - alpha
calibration_size = int(calib_size * len(labels))
CP_target = np.ceil(nominal_coverage*(calibration_size+1))/calibration_size
#############################
#find optimal sets and lambda
#############################
results = {
    "opt_b": {},
    "opt_lambda": {},
}

calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels = train_test_split(probs.numpy(), mean_probs.numpy(), labels, train_size=calib_size, random_state=exp_seed)

opt_lambda_BPS, _ = find_optimal_lambda(calib_probs, calib_labels, target=CP_target, batch_size=100)
opt_lambda_APS, _  = find_optimal_lambda(calib_mean_probs, calib_labels, target=CP_target, batch_size=100)

results["opt_lambda"]["BPS"] = opt_lambda_BPS
results["opt_lambda"]["BPS_cons"] = np.maximum(opt_lambda_BPS, nominal_coverage)
results["opt_lambda"]["BPS_nom"] = nominal_coverage
results["opt_lambda"]["APS"] = opt_lambda_APS
results["opt_lambda"]["APS_cons"] = np.maximum(opt_lambda_APS, nominal_coverage)
results["opt_lambda"]["APS_nom"] = nominal_coverage

for approach in results["opt_lambda"].keys(): 
    if approach[:3] == "BPS": 
        results["opt_b"][approach] = solve_b_in_batches(test_probs, results["opt_lambda"][approach], batch_size=100)
    elif approach[:3] == "APS": 
        results["opt_b"][approach] = solve_b_in_batches(test_mean_probs, results["opt_lambda"][approach], batch_size=100)


#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "seed " + exp_seed_str), exist_ok=True) 
with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, model, "calib_size "+calib_size_str, "alpha "+alpha_str, "seed " + exp_seed_str, "sets.pkl"), 'wb') as f:
    pickle.dump(results, f)
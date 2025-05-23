import sys
import os
import numpy as np
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from epiuc.conformal_prediction.uacp import solve_b_in_batches, find_optimal_lambda, all_boundary_points, tv
from epiuc.utils.general import calculate_quantile
from sklearn.model_selection import train_test_split


model = sys.argv[1]
alpha_str = sys.argv[2]
alpha = float(alpha_str)
alpha_credal_str = sys.argv[3]
alpha_credal = float(alpha_credal_str)
calib_size_str = sys.argv[4]
calib_size = float(calib_size_str)
exp_seed_str = sys.argv[5]
exp_seed = int(exp_seed_str)



data = "cifar10"
#############################
#Load predictions
#############################
with open(os.path.join(root_path, "all_results", data, model, "predictions.pkl"), 'rb') as f:
    predictions = pickle.load(f)
labels = predictions["labels"]
probs = predictions["probs"]
mean_probs = predictions["mean_probs"]

calibration_size = 0.5 * int(calib_size * len(labels))
nominal_coverage = 1 - alpha
CP_target = np.ceil(nominal_coverage*(calibration_size+1))/calibration_size

nominal_coverage_credal = 1 - alpha_credal
CP_target_credal = np.ceil(nominal_coverage_credal*(calibration_size+1))/calibration_size
#############################
#data splits
#############################
annotations_prob = np.load(os.path.join(root_path, "data", "CIFAR-10-H", "cifar10h-probs.npy"))
calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels, calib_annotations, test_annotations = train_test_split(probs.numpy(), mean_probs.numpy(), labels.numpy(), annotations_prob, train_size=calib_size, random_state=exp_seed)
calib_probs_credal, calib_probs_BPS, calib_mean_probs_credal, calib_mean_probs_BPS, calib_labels_credal, calib_labels_BPS, calib_annotations_credal, calib_annotations_BPS = train_test_split(calib_probs, calib_mean_probs, calib_labels, calib_annotations, train_size=0.5, random_state=exp_seed)



#############################
#construct credal sets
#############################
tv_scores = tv(calib_mean_probs_credal.squeeze(axis=1), calib_annotations_credal)
credal_quantile = calculate_quantile(tv_scores, 1-CP_target_credal)
Q_calib_BPS, _ = all_boundary_points(calib_mean_probs_BPS.squeeze(axis=1), np.repeat(credal_quantile,calib_mean_probs_BPS.shape[0]))
Q_test, _ = all_boundary_points(test_mean_probs.squeeze(axis=1), np.repeat(credal_quantile,test_mean_probs.shape[0]))

#############################
#find optimal sets and lambda
#############################
results = {
    "opt_b": {},
    "opt_lambda": {},
    "test_credal_cvg": tv(test_mean_probs.squeeze(axis=1), test_annotations) < credal_quantile
}

opt_lambda_BPS, _ = find_optimal_lambda(Q_calib_BPS, calib_labels_BPS, target=CP_target, batch_size=100)
opt_lambda_APS, _  = find_optimal_lambda(calib_mean_probs_BPS, calib_labels_BPS, target=CP_target, batch_size=100)

results["opt_lambda"]["BPS"] = opt_lambda_BPS
results["opt_lambda"]["BPS_cons"] = np.maximum(opt_lambda_BPS, nominal_coverage)
results["opt_lambda"]["BPS_nominal"] = nominal_coverage
results["opt_lambda"]["APS"] = opt_lambda_APS
results["opt_lambda"]["APS_cons"] = np.maximum(opt_lambda_APS, nominal_coverage)
results["opt_lambda"]["APS_nominal"] = nominal_coverage

for approach in results["opt_lambda"].keys(): 
    if approach[:3] == "BPS": 
        results["opt_b"][approach] = solve_b_in_batches(Q_test, results["opt_lambda"][approach], batch_size=100)
    elif approach[:3] == "APS": 
        results["opt_b"][approach] = solve_b_in_batches(test_mean_probs, results["opt_lambda"][approach], batch_size=100)


#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_results", data, "credal", model, "calib_size "+calib_size_str, "alpha credal "+alpha_credal_str, "alpha "+alpha_str, "seed " + exp_seed_str), exist_ok=True) 
with open(os.path.join(root_path, "all_results", data, "credal", model, "calib_size "+calib_size_str, "alpha credal "+alpha_credal_str,"alpha "+alpha_str, "seed " + exp_seed_str, "res.pkl"), 'wb') as f:
    pickle.dump(results, f)
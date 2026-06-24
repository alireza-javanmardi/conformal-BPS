import sys
import os
import pickle
root_path = os.path.abspath('..')
sys.path.append(root_path)
import numpy as np
from src.BPS import *
from src.helper import one_hot
from sklearn.model_selection import train_test_split


dataset = sys.argv[1]
model = sys.argv[2]
model_seed_str = sys.argv[3]
seed = int(model_seed_str)
gamma_str = sys.argv[4]
gamma = float(gamma_str)
alpha_cp_str = sys.argv[5]
alpha_cp = float(alpha_cp_str)
cnd_cvg_thr_str = sys.argv[6]
cond_cvg_thr = float(cnd_cvg_thr_str)
split_seed_str = sys.argv[7]
split_seed = int(split_seed_str)


if dataset == "cifar10":
    calib_size = 0.2
elif dataset == "chaosnli":
    calib_size = 0.5
elif dataset == "qualitymri":
    calib_size = 0.5

# exp_seed = 423
nominal_coverage = 1 - alpha_cp
results = {
    "sets": {},
    "opt_lambda": {},
}


with open(os.path.join(root_path, "all_credal_predictions", dataset, model, model_seed_str, gamma_str, "predictions.pkl"), 'rb') as f:
    predictions = pickle.load(f)


calib_credals, test_credals, calib_labels, test_labels, calib_true_probs, test_true_probs = train_test_split(predictions["outputs"], predictions["targets_y"].astype(int), predictions["targets_p"], train_size=calib_size, random_state=2026+split_seed)
batch_size = min(100, test_credals.shape[0])
BPS_no_calib = solve_b_in_batches(test_credals, cond_cvg_thr, batch_size=batch_size)


calibration_size = int(calib_size * len(predictions["outputs"]))
CP_target = np.ceil(nominal_coverage*(calibration_size+1))/calibration_size
# calibrate BPS with inverese risk = mean conditional satisfaction and first_order data
lambda_cond_cvg_satisfaction = lambda_optimizer(solve_b_in_batches, mean_cond_cvg_satisfaction,
                                    target=CP_target, tol=1e-5, lo=0.0, hi=1.0,
                                    set_predictor_kwargs={"credal_sets": calib_credals, "batch_size": batch_size}, 
                                    risk_kwargs={"true_dists": calib_true_probs, "desired_cond_cvg": cond_cvg_thr},
                                    return_sets=False)
BPS_cond_cvg_satisfaction = solve_b_in_batches(test_credals, lambda_cond_cvg_satisfaction, batch_size=batch_size)

# calibrate BPS with inverese risk = mean conditional satisfaction and zero_order data
lambda_cond_cvg_satisfaction_zero = lambda_optimizer(solve_b_in_batches, mean_cond_cvg_satisfaction,
                                    target=CP_target, tol=1e-5, lo=0.0, hi=1.0,
                                    set_predictor_kwargs={"credal_sets": calib_credals, "batch_size": batch_size}, 
                                    risk_kwargs={"true_dists": one_hot(calib_labels, K=calib_true_probs.shape[1]), "desired_cond_cvg": cond_cvg_thr},
                                    return_sets=False)
BPS_cond_cvg_satisfaction_zero = solve_b_in_batches(test_credals, lambda_cond_cvg_satisfaction_zero, batch_size=batch_size)

# calibrate BPS with inverese risk = mean conditional coverage and first_order data
lambda_mean_cond_cvg = lambda_optimizer(solve_b_in_batches, mean_cond_cvg,
                                    target=CP_target, tol=1e-5, lo=0.0, hi=1.0,
                                    set_predictor_kwargs={"credal_sets": calib_credals, "batch_size": batch_size}, 
                                    risk_kwargs={"true_dists": calib_true_probs},
                                    return_sets=False)
BPS_mean_cond_cvg= solve_b_in_batches(test_credals, lambda_mean_cond_cvg, batch_size=batch_size)

# calibrate BPS with inverese risk = marginal coverage and zero_order data
lambda_marg_cvg = lambda_optimizer(solve_b_in_batches, marg_cvg,
                                    target=CP_target, tol=1e-5, lo=0.0, hi=1.0,
                                    set_predictor_kwargs={"credal_sets": calib_credals, "batch_size": batch_size}, 
                                    risk_kwargs={"labels": calib_labels},
                                    return_sets=False)
BPS_marg_cvg= solve_b_in_batches(test_credals, lambda_marg_cvg, batch_size=batch_size)

#save lambdas and sets
results["sets"]["BPS_no_calib"] = BPS_no_calib
results["opt_lambda"]["BPS_cond_cvg_satisfaction"] = lambda_cond_cvg_satisfaction
results["sets"]["BPS_cond_cvg_satisfaction"] = BPS_cond_cvg_satisfaction
results["opt_lambda"]["BPS_cond_cvg_satisfaction_zero"] = lambda_cond_cvg_satisfaction_zero
results["sets"]["BPS_cond_cvg_satisfaction_zero"] = BPS_cond_cvg_satisfaction_zero
results["opt_lambda"]["BPS_mean_cond_cvg"] = lambda_mean_cond_cvg
results["sets"]["BPS_mean_cond_cvg"] = BPS_mean_cond_cvg
results["opt_lambda"]["BPS_marg_cvg"] = lambda_marg_cvg
results["sets"]["BPS_marg_cvg"] = BPS_marg_cvg

#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_sets", "unknown validity", dataset, model, model_seed_str, gamma_str), exist_ok=True) 
with open(os.path.join(root_path, "all_sets", "unknown validity", dataset, model, model_seed_str, gamma_str, f"alphaCP_{alpha_cp_str}_cond_{cond_cvg_thr}_seed_{split_seed}.pkl"), 'wb') as f:
    pickle.dump(results, f)
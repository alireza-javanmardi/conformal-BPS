import sys
import os
import pickle
root_path = os.path.abspath('..')
sys.path.append(root_path)
import numpy as np
from src.BPS import *
from src.helper import tv, compute_quantile, get_tv_elementary_extreme_points_batch
from sklearn.model_selection import train_test_split


dataset = sys.argv[1]
model = sys.argv[2]
model_seed_str = sys.argv[3]
seed = int(model_seed_str)
gamma_str = sys.argv[4]
gamma = float(gamma_str)
epsilon_credal_str = sys.argv[5]
epsilon_credal = float(epsilon_credal_str)
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
results = {}


with open(os.path.join(root_path, "all_credal_predictions", dataset, model, model_seed_str, gamma_str, "predictions.pkl"), 'rb') as f:
    predictions = pickle.load(f)


calib_credals, test_credals, calib_labels, test_labels, calib_true_probs, test_true_probs = train_test_split(predictions["outputs"], predictions["targets_y"].astype(int), predictions["targets_p"], train_size=calib_size, random_state=2026+split_seed)

tv_scores = tv(calib_credals.mean(axis=1), calib_true_probs)
credal_quantile = compute_quantile(tv_scores, epsilon_credal)

conformalized_credal_sets = get_tv_elementary_extreme_points_batch(test_credals.mean(axis=1), credal_quantile)

batch_size = min(100, test_credals.shape[0])
BPS_no_calib = solve_b_in_batches(conformalized_credal_sets, cond_cvg_thr, batch_size=batch_size)
APS_no_calib = solve_b_in_batches(test_credals.mean(axis=1, keepdims=True), cond_cvg_thr, batch_size=batch_size)

results["sets_BPS"] = BPS_no_calib
results["sets_APS"] = APS_no_calib
results["conformalized_credal_sets"] = conformalized_credal_sets
results["credal_quantile"] = credal_quantile

#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_sets", "partially valid", dataset, model, model_seed_str, gamma_str), exist_ok=True) 
with open(os.path.join(root_path, "all_sets", "partially valid", dataset, model, model_seed_str, gamma_str, f"epsilonCredal_{epsilon_credal_str}_cond_{cond_cvg_thr}_seed_{split_seed}.pkl"), 'wb') as f:
    pickle.dump(results, f)
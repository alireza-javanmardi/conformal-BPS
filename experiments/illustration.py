import sys
import os
import torch
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from torch.distributions import Dirichlet
from epiuc.conformal_prediction.uacp import all_boundary_points



torch.manual_seed(0)
n = 1000 #number of instances
K = 3 #number of classes
dir_alpha = torch.tensor(K*[1.0]) #unifor Dirichlet parameter
mean_probs = Dirichlet(dir_alpha).sample((n,)) #mean probabilities



def true_prob(prob, K, d_min, d_max): 
    """
    this function takes probability distribution and returns another one that has 
    a total variation distence in the range [d_min, d_max] with that
    
    """
    for _ in range(100000):
        dir_alpha = torch.tensor(K*[1.0]) 
        candidate_prob = Dirichlet(dir_alpha).sample()
        tv = 0.5 * (prob - candidate_prob).abs().sum() 
        if tv <= d_max and tv >= d_min:
            return candidate_prob, tv
    print("could not find any")
    return  prob, 0

#credal sets are defined as a total variation distance ball around mean_probs and the radius of d_credal
d_credal = [0.05, 0.1, 0.2, 0.3]

all_data = {"mean_probs":mean_probs, "probs":{}, "true_probs":{}, "tv_mean_true":{}}

for d in d_credal:
    probs, _ = all_boundary_points(mean_probs.numpy(), d)
    true_probs = []
    tv_mean_true = []
    for i  in range(n): 
        d_min = d*torch.rand(1)
        t_prob, tv_t_prob = true_prob(mean_probs[i], K, d_min=d_min, d_max=d)
        tv_mean_true.append(tv_t_prob)
        true_probs.append(t_prob)
    true_probs = torch.stack(true_probs)
    tv_mean_true = torch.stack(tv_mean_true)
    all_data["probs"][d] = probs
    all_data["true_probs"][d] = true_probs
    all_data["tv_mean_true"][d] = tv_mean_true

#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_results", "illustration"), exist_ok=True) 
with open(os.path.join(root_path, "all_results", "illustration", "res.pkl"), 'wb') as f:
    pickle.dump(all_data, f)
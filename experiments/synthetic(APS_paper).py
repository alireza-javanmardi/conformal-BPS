import sys
import os
import pickle
# Add root folder path to sys.path
root_path = os.path.abspath('..')
sys.path.append(root_path)

from sklearn.model_selection import train_test_split
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Evidential_Classifier, Ensemble_Classifier, MC_Classifier
from sklearn.metrics import accuracy_score
from torch.distributions import Dirichlet


import numpy as np
import random
import torch 

data_size = sys.argv[1]
rarity_param = sys.argv[2]
data_seed_str = sys.argv[3]
exp_seed = int(data_seed_str)


# Set the random seed for numpy
np.random.seed(exp_seed)
# Set the random seed for python's built-in random module
random.seed(exp_seed)

# Set the random seed for PyTorch
torch.manual_seed(exp_seed)
torch.cuda.manual_seed_all(exp_seed)  # for GPU (if you're using CUDA)
torch.backends.cudnn.deterministic = True  # for deterministic operations on CUDA
torch.backends.cudnn.benchmark = False 

def make_bootstrap_loader(n, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    # sample with replacement
    indices = torch.randint(high=n, size=(n,), generator=generator).tolist()
    return indices

#############################
#Data Generation 
#############################
n = int(data_size)
rarity = float(rarity_param)

X1 = np.random.binomial(1, rarity, size=n).reshape((-1,1))
X1[X1==0] = -8
X29 = np.random.normal(loc=0.0, scale=1.0, size=(n,9))
X = np.concatenate((X1,X29), axis=1)
beta = np.random.normal(loc=0.0, scale=1.0, size=(10,10))
Z = np.exp(np.matmul(X,beta))
W = Z/Z.sum(axis=1)[:,None]
y = []
for i in range(n):
    y.append(np.where(np.random.multinomial(1, W[i], size=None)==1)[0][0])
y = np.array(y)


# make X and y compatible with pytorch
X = X.astype(np.float32)
W = W.astype(np.float32)
y = y.astype(np.int64)

#############################
#train split
#############################
X_train, X_t, y_train, y_t, W_train, W_t = train_test_split(X, y, W, test_size=0.5, random_state=exp_seed)
n_predictors = 5
N_EPOCHS = 50
#############################
#training
#############################

probs = {}
mean_probs = {}
# This is the base net which will be loaded
net = MLP_Classifier(input_shape=X.shape[1], n_classes=10, n_layers=2, num_neurons=64, dropout_prob=0.3, batch_norm=False, random_state=exp_seed)

evidential_model = Evidential_Classifier(
    base_model=net,
    kl_reg_scaler=0.001,
    evidence_method="softplus",
    random_state=exp_seed,
)
evidential_model.fit(X_train, y_train, n_epochs=N_EPOCHS)
evi_alphas, _ = evidential_model.predict(X_t, raw_output=True)
probs["evidential"] = torch.stack([Dirichlet(dir_alpha).sample((5,)) for dir_alpha in evi_alphas])



ensemble_model = Ensemble_Classifier(
    base_model=net,
    n_models=n_predictors,
    random_state=exp_seed,
)
ensemble_model.fit(X_train, y_train, n_epochs=N_EPOCHS)
probs["ensemble"],_ = ensemble_model.predict(X_t, raw_output=True)


# As the net already contains dropout, we can use it as a MC dropout model
mc_model = MC_Classifier(
    base_model=net,
    n_iterations=n_predictors,
    random_state=exp_seed,
)
mc_model.fit(X_train, y_train, n_epochs=N_EPOCHS)
probs["mc"],_ = mc_model.predict(X_t, raw_output=True)
mean_probs["mc"] = probs["mc"].mean(axis=1).unsqueeze(1)

BS_probs = []
BS_ensemble = []
for m in range(n_predictors):
    net = MLP_Classifier(input_shape=X.shape[1], n_classes=10, n_layers=2, num_neurons=64, dropout_prob=0.3, batch_norm=False, random_state=exp_seed)
    indices = make_bootstrap_loader(X_train.shape[0], batch_size=64, seed=exp_seed+m)
    net = net.fit(X_train[indices], y_train[indices], n_epochs=N_EPOCHS)
    BS_probs.append(net.predict(X_t))
    BS_ensemble.append(net)

probs["BS ensemble"] = torch.stack(BS_probs,axis=1)

results = {"X_t": X_t,
           "y_t": y_t, 
           "W_t": W_t, 
           "probs": probs}

#############################
#save results
#############################
os.makedirs(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str), exist_ok=True) 
with open(os.path.join(root_path, "all_results", "synthetic", "data_size " + data_size, "rarity  "+rarity_param, "data seed " + data_seed_str, "res.pkl"), 'wb') as f:
    pickle.dump(results, f)

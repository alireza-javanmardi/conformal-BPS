import os
import pickle
import sys
import torch
root_path = os.path.abspath('..')
sys.path.append(root_path)
from src.real_data_loader import get_test_loader
from src.real_model_loader import get_model, LikelihoodEnsemble, DesterckeEnsemble, load_ensemble, torch_get_outputs_destercke, torch_get_outputs



dataset = sys.argv[1]
model = sys.argv[2]
seed_str = sys.argv[3]
seed = int(seed_str)
gamma_str = sys.argv[4]
gamma = float(gamma_str)



test_loader = get_test_loader(dataset, seed, batch_size=128)

if dataset == "cifar10":
    classes = 10
    base = 'resnet'
elif dataset == "chaosnli":
    classes = 3 
    base = 'fcnet'
elif dataset == "qualitymri":
    classes = 2
    base = 'torchresnet'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = 'cpu'
# device = torch.device('cpu')

n_members=20 
tobias = 100
model_path = f'{os.path.join(root_path, "saved_models")}'

if model == "RL_credal":
    ensemble = LikelihoodEnsemble(get_model(base, classes), classes, n_members=n_members, tobias_value=tobias)
    load_ensemble(ensemble, f'{model_path}/{dataset}_{base}_{n_members}_{tobias}_{gamma}_{seed}')
    ensemble = ensemble.to(device)
    ensemble.eval()
    inputs, targets_y, targets_p, outputs = torch_get_outputs(ensemble, test_loader, device)

elif model == "destercke":
    ensemble = DesterckeEnsemble(get_model(base, classes), n_members)
    ensemble.load_state_dict(torch.load(f'{model_path}/baseline_{dataset}_credalensembling_{n_members}_{seed}'))
    ensemble = ensemble.to(device)
    ensemble.eval()
    inputs, targets_y, targets_p, outputs = torch_get_outputs_destercke(ensemble, test_loader, device, gamma)

else:
    raise ValueError("model should be either RL_credal or destercke")

predictions = {
    "inputs": inputs.detach().cpu().numpy(),
    "targets_y": targets_y.detach().cpu().numpy(),
    "targets_p": targets_p.detach().cpu().numpy(),
    "outputs": outputs.detach().cpu().numpy(),
}

os.makedirs(os.path.join(root_path, "all_credal_predictions", dataset, model, seed_str, gamma_str), exist_ok=True) 
with open(os.path.join(root_path, "all_credal_predictions", dataset, model, seed_str, gamma_str, "predictions.pkl"), 'wb') as f:
    pickle.dump(predictions, f)






import torch

# Data Loading
ROOT_PATH = "data/"
BASE_PATH = ROOT_PATH + "datasets/"

# Optimizer
SGD_OPTIMIZER_CONFIGS = {
    "optimizer": torch.optim.SGD,
    "params": {
        "lr": 1e-3,
        "weight_decay": 0,
        "momentum": 0.9,
        "nesterov": True,
    },
}
ADAM_OPTIMIZER_CONFIGS = {
    "optimizer": torch.optim.Adam,
    "params": {"lr": 1e-3, "weight_decay": 0},
}
OPTIMIZER_CONFIGS = {
    "sgd": SGD_OPTIMIZER_CONFIGS,
    "adam": ADAM_OPTIMIZER_CONFIGS,
}

# Scheduler
STEP_SCHEDULER_CONFIGS = {
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "params": {
        "step_size": 100,
        "gamma": 0.1,
    },
}

COSINE_SCHEDULER_CONFIGS = {
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "params": {
        "T_max": 50,
        "eta_min": 0,
    },
}

COSINE_WARMUP_SCHEDULER_CONFIGS = {
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "params": {
        "T_0": 50,
        "T_mult": 1,
        "eta_min": 0,
    },
}
SCHEDULER_CONFIGS = {
    "step": STEP_SCHEDULER_CONFIGS,
    "cosine": COSINE_SCHEDULER_CONFIGS,
    "cosine_warmup": COSINE_WARMUP_SCHEDULER_CONFIGS,
}


DEFAULT_CONFIGS = {
    "backend": "cudagraphs",
    "device": "cuda",
    "dtype": "float32",
    "seed": 42,
    "verbose": False,
    "n_epochs": 100,
    "n_layers": 3,
    "num_neurons": 100,
    "input_shape": 1,
    "dropout_rate": 0.1,
}
BACKEND = DEFAULT_CONFIGS["backend"]
DATALOADER_CONFIGS = {
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 0,
    "pin_memory": True,
}

IMAGENET_C_PERTUBATIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]
CIFAR_C_PERTUBATIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

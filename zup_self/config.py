"""Hyperparameter presets and DictBlock CSC configuration."""
import torch.nn as nn
import torch

CSC_CONFIG = {
    "MU": 0.0,
    "SQUARE_NOISE": True,
    "EXPANSION_FACTOR": 1,
    "NONEGATIVE": True,
    "NUM_LAYERS": 2,
    "WNORM": True,
}

DATASET_PRESETS = {
    "WV3": {
        "sensor": "WV3",
        "path": "/media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5",
    },
    "WV2": {
        "sensor": "WV2",
        "path": "/media/zouhe/Elements/Data/PanCollection/test_data/test_wv2_OrigScale_multiExm1.h5",
    },
    "QB": {
        "sensor": "QB",
        "path": "/media/zouhe/Elements/Data/PanCollection/test_data/test_qb_OrigScale_multiExm1.h5",
    },
    "GF2": {
        "sensor": "WV4",
        "path": "/media/zouhe/Elements/Data/PanCollection/test_data/test_gf2_OrigScale_multiExm1.h5",
    },
}

TRAIN_DEFAULTS = {
    "lr_phase1": 0.015,
    "epochs_phase1": 240,
    "lr_phase2": 0.01,
    "epochs_phase2": 2000,
    "batch_size": 1,
    "w_spa": 200,
    "w_spec": 250,
    "w_var": 1_850_000.0,
    "lmbd_weight": 500.0,
    "temperature": 1.0,
    "ratio": 4,
    "seed": 10,
    "max_value": 2047.0,
}

ABLATION_PRESETS = {}
for t in [3, 5, 7]:
    ABLATION_PRESETS[f"fista_T{t}"] = {"NUM_LAYERS": t}
for v in [10, 100, 200, 300, 400, 500]:
    ABLATION_PRESETS[f"lmbd_weight_{v}"] = {"lmbd_weight": v}


def get_all_lmbd_values(model):
    """Collect all lmbd parameter values from DictBlock modules."""
    values = []
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            values.append(m.lmbd.item())
    return values


def compute_lmbd_regularization(model):
    """Compute L2 regularization over all lmbd parameters."""
    total = 0.0
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            total = total + m.lmbd.pow(2).sum()
    return total


def freeze_lmbd(model, value=0.1):
    """Fix all DictBlock lmbd parameters to a constant value."""
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            m.lmbd = nn.Parameter(torch.tensor([value], device=m.lmbd.device), requires_grad=False)


def clamp_lmbd(model, min_val=0.0, max_val=10.0):
    """Clamp all DictBlock lmbd to [min_val, max_val]."""
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            m.lmbd.data.clamp_(min=min_val, max=max_val)


def set_fista_steps(model, n_steps):
    """Override FISTA iteration count in all DictBlocks."""
    for m in model.modules():
        if hasattr(m, 'n_steps'):
            m.n_steps = n_steps

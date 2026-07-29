from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

root_path = Path(__file__).parent.parent
model_path = root_path / "original_ann" / "output" / "model.pt"
pattern_path = root_path / "original_ann" / "patterns" / "pattern.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 999  # original training seed (paper / git history)
torch.manual_seed(SEED)
np.random.seed(SEED)

D_in, H1, H2, D_out = 6, 500, 500, 3


def new_model():
    model = nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        # nn.Tanh(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        # nn.Tanh(),
        nn.Linear(H2, D_out),
    ).to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    return model


def load_model():
    model = new_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

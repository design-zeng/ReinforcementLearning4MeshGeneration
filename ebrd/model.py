from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


base_path = Path(__file__).parent.parent
domains_path = base_path / "samples" / "domains"
output_path = base_path / "ebrd" / "output"
augmentation_path = output_path / "data_augmentation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 999  # original training seed (paper / git history)
torch.manual_seed(SEED)
np.random.seed(SEED)


class FNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_space = 18
        self.action_space = 2
        self.type_space = 3

        self.fc1 = nn.Linear(self.state_space, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.action_head = nn.Linear(16, self.action_space)
        self.type_head = nn.Linear(16, self.type_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.action_head(x), self.type_head(x)


def get_action(state, model):
    state = torch.FloatTensor(state).to(device)
    action, type_values = model(state)
    return action.tolist(), float(torch.argmax(type_values) / 2)


def load_model(model_path):
    model = FNNPolicy().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

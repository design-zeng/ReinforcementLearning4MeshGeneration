import json
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from gym_env import Gym_Env


base_path = Path(__file__).parent.parent

TIMESTEPS = 200000
DOMAINS = ["b15", "basic", "basic1", "bird", "bird1", "boundary4"]

boundary = np.array(json.load(open(base_path / "samples" / "domains" / f"{DOMAINS[0]}.json", "r")))
environment = Gym_Env(boundary)
model = SAC("MlpPolicy", environment)

for domain in DOMAINS:
    print(domain)
    boundary = np.array(json.load(open(base_path / "samples" / "domains" / f"{domain}.json", "r")))
    environment = Gym_Env(boundary)
    model.set_env(environment)
    model.learn(TIMESTEPS)

os.makedirs(base_path / "sac_ebd" / "output", exist_ok=True)
model.save(base_path / "sac_ebd" / "output" / "model.zip")

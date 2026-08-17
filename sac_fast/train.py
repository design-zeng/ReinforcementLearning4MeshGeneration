import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from sac_fast.gym_env import Gym_Env


base_path = Path(__file__).parent.parent

DOMAINS = ["b15", "basic", "basic1", "bird", "bird1", "boundary4"]


if __name__ == "__main__":
    TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

    boundary = np.array(json.load(open(base_path / "samples" / "domains" / f"{DOMAINS[0]}.json", "r")))
    environment = Gym_Env(boundary)
    model = SAC("MlpPolicy", environment)

    for domain in DOMAINS:
        print(domain)
        boundary = np.array(json.load(open(base_path / "samples" / "domains" / f"{domain}.json", "r")))
        environment = Gym_Env(boundary)
        model.set_env(environment)
        model.learn(TIMESTEPS)

    os.makedirs(base_path / "sac_fast" / "output", exist_ok=True)
    model.save(base_path / "sac_fast" / "output" / "model.zip")

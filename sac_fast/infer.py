import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from sac_fast.gym_env import Gym_Env


base_path = Path(__file__).parent.parent

DOMAIN = "basic1"
boundary = np.array(json.load(open(base_path / "samples" / "domains" / f"{DOMAIN}.json", "r")))
environment = Gym_Env(boundary)
obs, info = environment.reset()

model = SAC.load(base_path / "sac_fast" / "output" / "model.zip")

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = environment.step(action)
    done = terminated or truncated
    if done:
        for i, j in environment.mesh.edges:
            x = (environment.mesh.vertices[i][0], environment.mesh.vertices[j][0])
            y = (environment.mesh.vertices[i][1], environment.mesh.vertices[j][1])
            plt.plot(x, y, "k-")
        plt.show()

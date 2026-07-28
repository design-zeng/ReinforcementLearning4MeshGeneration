import os
from pathlib import Path

import torch
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

from general.polygon_generators import read_polygon
from general.boundary_env import BoudaryEnv
from sac.custom_callback import CustomCallback

base_path = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version = 77 ## 46 for ppo
method_name = 'sac'

method_settings = {
    "a2c": (A2C,  {
        "policy_kwargs": {
            "activation_fn" : torch.nn.ReLU,
            "net_arch": [64, {"pi": [32, 32], "vf": [32, 32]}]
        },
        "seed": 999,
        "learning_rate": 3e-4,
    }),
    "ddpg": (DDPG, {
        "policy_kwargs": {
            "activation_fn" : torch.nn.ReLU,
            "net_arch": [256, 256]
        },
        "seed": 999,
        "learning_rate": 3e-4,
    }),
    "ppo": (PPO, {
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [{"pi": [128, 128], "vf": [128, 128]}]
        },
        "seed": 111,
        "learning_rate": 3e-4,
        "gamma": 0.5,
        "device": device
    }),
    "sac": (SAC, {
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [128, 128, 128] # alt: 32,128,128,128,64,32
        },
        "seed": 999,
        "learning_rate": 3e-4,
        "learning_starts": 10000,
        "batch_size": 100,
        "device": device
    }),
    "td3": (TD3, {
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [256, 256]
        },
        "seed": 999,
        "learning_rate": 3e-4,
        "learning_starts": 10000
    })
}

environments = [
    [
        1500000,
        BoudaryEnv(read_polygon(base_path / "samples" / "domains" / "random1_1.json")),
        BoudaryEnv(read_polygon(base_path / "samples" / "domains" / "random1_1.json"))
    ]
]

def train_all():
    Algo, kwargs = method_settings[method_name]

    for i, (total_timesteps, train_env, eval_env) in enumerate(environments):
        output_path = base_path / "sac" / "output" / "logs" / method_name / f"{version}" / f"{i}"
        os.makedirs(output_path, exist_ok=True)

        if i == 0:
            model = Algo("MlpPolicy", train_env, tensorboard_log=output_path, **kwargs)
        else:
            previous_model_path = base_path / "sac" / "output" / "logs" / method_name / f"{version}" / f"{i - 1}" / "model.zip"
            model = Algo.load(previous_model_path, env=train_env)

        eval_callback = CustomCallback(eval_env, best_model_save_path=output_path,
                                           log_path=output_path, eval_freq=1000,
                                           n_eval_episodes=1,
                                           deterministic=False, render=False)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(output_path / "model.zip")

if __name__ == "__main__":
    train_all()

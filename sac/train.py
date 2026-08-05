import os
from pathlib import Path

import torch
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

from general.polygon_reader import read_polygon
from general.boundary_env import BoudaryEnv
from sac.custom_callback import CustomCallback


base_path = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


version = 77 # change if wish to preserve old training data

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
            "net_arch": [128, 128, 128]
            # [32, 32], [64, 64, 64, 64, 64], [32, 128, 128, 128, 64, 32]
        },
        "seed": 999, # 356, 567
        "learning_rate": 3e-4,
        "learning_starts": 10000,
        "batch_size": 100, #256
        "device": device,
        #"gamma": 0.99,
        #"gradient_steps": 1,
        #"tau": 5e-3
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


def train(method_name):
    Algo, kwargs = method_settings[method_name]

    for i, (total_timesteps, train_env, eval_env) in enumerate(environments):

        current_output_path = base_path / "sac" / "output" / "logs" / method_name / f"{version}" / f"{i}"
        os.makedirs(current_output_path, exist_ok=True)

        if i == 0:
            model = Algo("MlpPolicy", train_env, tensorboard_log=current_output_path, **kwargs)
        else:
            previous_model_path = base_path / "sac" / "output" / "logs" / method_name / f"{version}" / f"{i - 1}" / "model.zip"
            model = Algo.load(previous_model_path, env=train_env)

        eval_callback = CustomCallback(eval_env, current_output_path, current_output_path)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        current_model_path = current_output_path / "model.zip"
        model.save(current_model_path)


if __name__ == "__main__":
    # method_name = "a2c"
    # train(method_name)
    # method_name = "ddpg"
    # train(method_name)
    # method_name = "ppo"
    # train(method_name)
    method_name = "sac"
    train(method_name)
    # method_name = "td3"
    # train(method_name)
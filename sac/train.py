import os
from pathlib import Path
import configparser

import torch
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sac.polygon_generators import read_polygon
from sac.boundary_env import BoudaryEnv
from sac.custom_callback import CustomCallback

base_path = Path(__file__).parent.parent.parent
config = configparser.ConfigParser()
config.read(f'{base_path}/config')

device = torch.device("cuda:0")

o_env = BoudaryEnv(read_polygon(config['domains']['dolphine3'])) #dolphine3 basic2
eval_env = BoudaryEnv(read_polygon(config['domains']['dolphine3']))


version = '77' ## 46 for ppo
total_timesteps = 4000000
seed = 999 # change from 999 to 111
rl_method = 'sac'
learning_rate = 3e-4
# gamma = 0.5

parameter_tuning = {
    'gamma': 0.5
}

def prepare_eval_envs():
    domains = [
               [1500000, BoudaryEnv(read_polygon('D:/python_projects/meshgeneration/ui/domains/random1_1.json')),
                BoudaryEnv(read_polygon('D:/python_projects/meshgeneration/ui/domains/random1_1.json'))],
               ]
    return domains


def curriculum_learning(method_name):
    envs = prepare_eval_envs()
    for i, (learning_step, env, e_env) in enumerate(envs):
        if i <= 0:
            model_path = None
            # continue
        else:
            model_path = f"{config['default'][f'{method_name}_log']}/{version}/curriculum/{i - 1}/mesh"
        mesh_learning(method_name, env, e_env, learning_step,
                      model_path=model_path,
                      log_path=f"{config['default'][f'{method_name}_log']}/{version}/curriculum/{i}",
                      tensorboard_log=f"{config['default'][f'{method_name}_tensorboard_log']}/{version}/curriculum/{i}")


def mesh_learning(method_name, o_env, eval_env, total_timesteps, model_path=None,
                  log_path=None,
                  tensorboard_log=None):
    os.makedirs(log_path)
    os.makedirs(tensorboard_log)

    eval_callback = CustomCallback(eval_env, best_model_save_path=log_path,
                                   log_path=log_path, eval_freq=1000,
                                   n_eval_episodes=1,
                                   version=version,
                                   deterministic=False, render=False)

    # env = Monitor(o_env, f"{config['default']['a2c_log']}/{version}/")
    env = o_env
    if method_name == 'a2c':
        # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                      net_arch=[64, dict(pi=[32, 32], vf=[32, 32])])
        if model_path is not None:
            pass
        else:
            model = A2C(
                'MlpPolicy',
                env,
                # policy_kwargs=policy_kwargs,
                seed=seed,
                # learning_rate=learning_rate,
                tensorboard_log=tensorboard_log
            )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save the agent
        model.save(f"{log_path}/mesh")

        del model
        # the policy_kwargs are automatically loaded
        model = A2C.load(f"{log_path}/mesh", env=o_env)

    elif method_name == 'ddpg':
        if model_path is not None:
            pass
        else:
            # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
            #                      net_arch=[256, 256])
            model = DDPG(
                'MlpPolicy',
                env,
                seed=seed,
                # learning_rate=learning_rate,
                # policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log
            )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save the agent
        model.save(f"{log_path}/mesh")

        del model
        # the policy_kwargs are automatically loaded
        model = DDPG.load(f"{log_path}/mesh", env=o_env)

    elif method_name == 'ppo':
        if model_path is not None:
            model = PPO.load(model_path, env=o_env)
        else:
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[dict(pi=[128, 128], vf=[128, 128])])
            model = PPO(
                'MlpPolicy',
                env,
                seed=seed,
                # gamma=gamma,
                # learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                device=device
            )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save the agent
        model.save(f"{log_path}/mesh")

        # del model  # remove to demonstrate saving and loading

    elif method_name == 'sac':
        if model_path is not None:
            model = SAC.load(model_path, env=o_env)
        else:
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[128, 128, 128])#32, 128, 128, 128, 64, 32

            model = SAC(
                'MlpPolicy', # MlpPolicy
                env,
                seed=seed,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                learning_starts=10000,
                batch_size=100,
                tensorboard_log=tensorboard_log,
                device=device
            )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save the agent
        model.save(f"{log_path}/mesh")

        #
        # model = SAC.load(f"{log_path}/mesh", env=o_env)

    elif method_name == 'td3':
        if model_path is not None:
            model = TD3.load(model_path, env=o_env)
        else:
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[256, 256])
            model = TD3(
                'MlpPolicy',
                env,
                seed=seed,
                learning_starts=10000,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log
            )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save the agent
        model.save(f"{log_path}/mesh")

        # del model  # remove to demonstrate saving and loading
        #
        # model = TD3.load(f"{log_path}/mesh", env=o_env)

    # evaluation(model)
    return model


def evaluation(model, is_render=True):
    obs = o_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = o_env.step(action)
        if is_render:
            o_env.render()

# Create policy with a small network

# Use traditional actor-critic policy gradient updates to
# find good initial parameters
# checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
#                                          name_prefix='rl_model',
#                                          )
# model.learn(total_timesteps=100000, callback=checkpoint_callback)

# mesh_learning(rl_method, o_env, eval_env, total_timesteps,
#               f"{config['default'][f'{rl_method}_log']}/{version}/mesh",
#               tensorboard_log=f"{config['default'][f'{method_name}_tensorboard_log']}/{version}/")
curriculum_learning(rl_method)

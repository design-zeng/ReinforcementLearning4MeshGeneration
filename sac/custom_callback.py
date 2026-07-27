import os
import warnings
import configparser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import base_class

from sac.dummy_vec_env import DummyVecEnv

base_path = Path(__file__).parent.parent.parent
config = configparser.ConfigParser()
config.read(f'{base_path}/config')


def evaluate_policy(
    log_path: '',
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    version: int = 1,
    num_freq: int = 1
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards, episode_lengths = [], []
    not_reseted = True
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            # obs = env.reset()
            not_reseted = False
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        if render:
            env.close()
        if is_monitor_wrapped:
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # env.envs[0].boundary.savefig(f"{config['default'][f'{type(model).__name__.lower()}_log']}/{version}/ \
        #     {num_freq}_{len(episode_rewards)}.png", style='k-')
        env.envs[0].save_meshes(f"{log_path}/{num_freq}_{len(episode_rewards)}.png",
                                meshes=env.envs[0].generated_meshes,
                                indexing=True, style='k-', dpi=30)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class CustomCallback(EventCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        version: int = 1,
        check_freq: int = 100
    ):
        super(CustomCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.version = version
        self.check_freq = check_freq

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        # if log_path is not None:
        #     log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        info = locals_["info"]
        # VecEnv: unpack
        if not isinstance(info, dict):
            info = info[0]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.log_path,
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                version=self.version,
                num_freq=self.n_calls // self.eval_freq
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            self.model.save(os.path.join(self.log_path, f"{self.n_calls // self.eval_freq}"))

        # if self.n_calls > self.check_freq:
        #
        #     # Retrieve training reward
        #     x, y = ts2xy(load_results(self.log_path), 'timesteps')
        #     if len(x) > 0:
        #         # Mean training reward over the last 100 episodes
        #         mean_reward = np.mean(y[-100:])
        #         # if self.verbose > 0:
        #         #     print(f"Num timesteps: {self.num_timesteps}")
        #         #     print(
        #         #         f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
        #         self.logger.record("learn/mean_reward", mean_reward)
        #         self.logger.record("learn/reward", y[-1])

                # New best model, you could save the agent here
                # if mean_reward > self.best_mean_reward:
                #     self.best_mean_reward = mean_reward
                #     # Example for saving best model
                #     if self.verbose > 0:
                #         print(f"Saving new best model to {self.save_path}.zip")
                #     self.model.save(self.save_path)

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        if self.callback:
            self.callback.update_locals(locals_)
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from general.mesh_plotting import save_meshes


class CustomCallback(BaseCallback):

    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=1000, n_eval_episodes=1, deterministic=False, render=False, verbose=1):
        super().__init__(verbose)

        self.eval_env = eval_env

        self.best_model_save_path = best_model_save_path
        self.log_path = log_path

        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.deterministic = deterministic

        self.render = render

        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        os.makedirs(self.best_model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def _run_episode(self, num_freq: int, episode_idx: int) -> float:
        obs = self.eval_env.reset()

        done = False
        episode_reward = 0.0

        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, _ = self.eval_env.step(action)
            episode_reward += reward

            if self.render:
                self.eval_env.render()

        save_meshes(self.eval_env, f"{self.log_path}/{num_freq}_{episode_idx}.png",
                    meshes=self.eval_env.generated_meshes,
                    indexing=True, style='k-', dpi=30)
        return episode_reward

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        num_freq = self.n_calls // self.eval_freq
        rewards = [self._run_episode(num_freq, i + 1) for i in range(self.n_eval_episodes)]
        mean_reward = float(np.mean(rewards))

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {np.std(rewards):.2f}")

        self.logger.record("eval/mean_reward", mean_reward)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        self.model.save(os.path.join(self.log_path, f"{num_freq}"))

        return True

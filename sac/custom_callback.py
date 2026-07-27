import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """Periodically evaluate the policy, save a figure of each generated mesh, and
    checkpoint the model.

    Every ``eval_freq`` steps the current policy is rolled out on ``eval_env`` for
    ``n_eval_episodes`` episodes. The environment is driven directly (no VecEnv
    wrapper): unlike SB3's ``DummyVecEnv`` it does not auto-reset on ``done``, so the
    terminal mesh survives and can be rendered straight from ``eval_env``. After each
    evaluation the model is checkpointed to ``<log_path>/<eval_index>`` (these are what
    ``infer.py`` loads), and the best model so far is saved to
    ``<best_model_save_path>/best_model``.
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
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
        for path in (self.best_model_save_path, self.log_path):
            if path is not None:
                os.makedirs(path, exist_ok=True)

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
        # reset() is the only thing that clears the mesh, so the terminal state is
        # still intact here and can be rendered directly.
        self.eval_env.save_meshes(f"{self.log_path}/{num_freq}_{episode_idx}.png",
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
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"episode_reward={mean_reward:.2f} +/- {np.std(rewards):.2f}")
        self.logger.record("eval/mean_reward", mean_reward)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        self.model.save(os.path.join(self.log_path, f"{num_freq}"))
        return True

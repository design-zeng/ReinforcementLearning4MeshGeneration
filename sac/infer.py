import os
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

from general.polygon_reader import read_polygon
from general.boundary_env import BoudaryEnv

# sns.set_theme(style="darkgrid")

base_path = Path(__file__).parent.parent
domains_path = base_path / "samples" / "domains"          # the only user-provided inputs
output_path = base_path / "sac" / "output"
logs_path = output_path / "logs"                          # checkpoints + TensorBoard written by train.py: logs/<method>/<version>/<stage>/
eval_path = output_path / "evaluation"                    # figures, .inp meshes, samples produced here
experiments_path = output_path / "experiments"           # history-info produced here

# Must match the run produced by sac/train.py (its `version` and curriculum stage).
version = "77"
stage = "0"
method = "sac"


def prepare_eval_envs():
    names = ["boundary15", "random1_1", "random2_2"]  # easy / medium / hard
    return [BoudaryEnv(read_polygon(domains_path / f"{n}.json"), experiment_version=version, env_name=i)
            for i, n in enumerate(names)]


def prepare_model(method_name, model_path, env):
    loaders = {'a2c': A2C, 'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3}
    return loaders[method_name].load(model_path, env=env)


def find_checkpoints():
    """Auto-discover the checkpoints train.py wrote for (method, version, stage).

    Uses the numbered per-evaluation checkpoints, falling back to best_model.
    """
    ckpt_dir = logs_path / method / version / stage
    numbered = sorted((p for p in ckpt_dir.glob("*.zip") if p.stem.isdigit()),
                      key=lambda p: int(p.stem))
    if numbered:
        return numbered
    best = ckpt_dir / "best_model.zip"
    return [best] if best.exists() else []


def evaluation(is_render=False, deterministic=False, indexing=False, save_fig=False, save_samples=False):
    os.makedirs(eval_path / version, exist_ok=True)
    os.makedirs(experiments_path / version, exist_ok=True)
    is_random = 'T' if deterministic else 'F'

    checkpoints = find_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints under {logs_path / method / version / stage}. "
            f"Run `python -m sac.train` first (it writes them there).")

    envs = prepare_eval_envs()
    results = {c.stem: {'completed': [], 'n_elements': [], 'n_complete': 0} for c in checkpoints}

    for ckpt in checkpoints:
        for i, env in enumerate(envs):
            print(f'Evaluating checkpoint {ckpt.stem} on env {i}')
            model = prepare_model(method, ckpt, env)
            obs = env.reset()
            start = time.time()
            while True:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, info = env.step(action)
                if is_render:
                    env.render()
                if dones:
                    break
            print(f'Meshing running time: {time.time() - start}s')
            env.close()

            results[ckpt.stem]['completed'].append(info['is_complete'])
            results[ckpt.stem]['n_elements'].append(len(env.generated_meshes))
            results[ckpt.stem]['n_complete'] += 1 if info['is_complete'] else 0

            tag = f"{method}_{ckpt.stem}_env_{i}_{is_random}"
            if info['is_complete']:
                env.smooth(env.boundary.vertices)
            if save_fig and env.generated_meshes:
                env.save_meshes(eval_path / version / f"{tag}.png", meshes=env.generated_meshes,
                                quality=False, type=4, indexing=indexing, style='k-')
                # produced meshes -> consumed by the quality tools (general/tools/*)
                env.write_generated_elements_2_file(eval_path / version / f"{tag}.inp")
                # produced rule history -> consumed by sac/tools/rule_frequency_analysis.py
                env.save_history_info(experiments_path / version / f"{tag}_history_info")
                q = [env.get_quality(env.generated_meshes[j], 4) for j in range(len(env.generated_meshes))]
                print(f"element quality mean/std: {np.mean(q):.3f} / {np.std(q):.3f}")
            if save_samples and env.generated_meshes:
                samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4)
                env.save_samples(eval_path / version / f"{tag}.json",
                                 {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)

    with open(eval_path / version / 'evaluation.txt', 'w') as outfile:
        json.dump(results, outfile)
    print(f"Wrote results + meshes + history to {eval_path / version} and {experiments_path / version}")


def replication_evaluation(is_render=False, deterministic=False, indexing=False, save_fig=False, save_samples=False):
    """Mesh one domain many times to gather statistics (paper replication study)."""
    os.makedirs(eval_path / version, exist_ok=True)
    ckpt = logs_path / method / version / stage / "best_model.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run `python -m sac.train` first.")
    env = BoudaryEnv(read_polygon(domains_path / "boundary_fly_r2.json"))
    model = prepare_model(method, ckpt, env)
    times = 10

    results = {'sac': {'completed': [], 'n_elements': [], 'n_complete': 0}}
    for j in range(times):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(action)
            if is_render:
                env.render()
            if dones:
                break
        env.close()
        results['sac']['completed'].append(info['is_complete'])
        results['sac']['n_elements'].append(len(env.generated_meshes))
        results['sac']['n_complete'] += 1 if info['is_complete'] else 0
        if save_fig and info['is_complete']:
            env.smooth(env.boundary.vertices)

    with open(eval_path / version / "evaluation_repli.txt", 'w') as outfile:
        json.dump(results, outfile)


def element_number_box_plot():
    """Element-count-by-density box plot (paper figure). The three density panels
    are read from meshes produced by evaluation() (eval_path/<version>)."""
    data = pd.DataFrame({"Element number": [], "model": []})
    sparse = pd.DataFrame({"model": ['Sparse\ndensity'] * 10, "Element number": [69, 92, 72, 92, 84, 77, 78, 74, 85, 80]})
    medium = pd.DataFrame({"model": ['Medium\ndensity'] * 10, "Element number": [100, 95, 95, 108, 94, 103, 95, 97, 100, 100]})
    dense = pd.DataFrame({"model": ['Dense\ndensity'] * 10, "Element number": [165, 128, 170, 187, 156, 131, 146, 167, 136, 138]})
    data = pd.concat([data, sparse, medium, dense])

    panels = sorted((eval_path / version).glob("*.png"))[:3]
    fig = plt.figure(figsize=(6, 6))
    grid = GridSpec(2, 2)
    titles = ['(a) Sparse density', '(b) Medium density', '(c) Dense density']
    for k, cell in enumerate([grid[:1, :1], grid[:1, 1:], grid[1:, :1]]):
        ax = fig.add_subplot(cell)
        ax.grid(False); ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k])
        if k < len(panels):
            ax.imshow(mpimg.imread(panels[k]))

    box = fig.add_subplot(grid[1:, 1:])
    ax = sns.boxplot(ax=box, x="model", y="Element number", data=data)
    ax.set_title('(d) Element numbers for each density')
    ax.xaxis.label.set_visible(False)
    fig.tight_layout()
    plt.show()


def final_mesh(domain="boundary6",
               ckpt=logs_path / method / "78" / "1" / "best_model.zip",
               attempts=60):
    """Mesh one domain with the trained policy until it fully completes, then save
    the finished mesh. One command in, one finished mesh out."""
    os.makedirs(eval_path, exist_ok=True)
    env = BoudaryEnv(read_polygon(domains_path / f"{domain}.json"))
    model = prepare_model(method, ckpt, env)   # loaded once so the stochastic retries differ
    out = eval_path / f"final_mesh_{domain}.png"
    for k in range(attempts):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, info = env.step(action)
            if dones:
                break
        coverage = sum(m.compute_area()[0] for m in env.generated_meshes) / env.original_area
        if info['is_complete']:
            env.smooth(env.boundary.vertices)
            env.save_meshes(out, meshes=env.generated_meshes, quality=False, type=4, style='k-')
            print(f"Completed {domain} on attempt {k + 1} "
                  f"({len(env.generated_meshes)} elements, {coverage * 100:.0f}% area). Saved {out}")
            return
    env.save_meshes(out, meshes=env.generated_meshes, quality=False, type=4, style='k-')
    print(f"No full completion in {attempts} attempts; saved best-effort partial to {out}")


if __name__ == '__main__':
    # One command -> one finished mesh (retries until the domain fully meshes).
    final_mesh()
    # evaluation(is_render=False, deterministic=False, indexing=False, save_fig=True, save_samples=False)
    # replication_evaluation(save_fig=True)
    # element_number_box_plot()

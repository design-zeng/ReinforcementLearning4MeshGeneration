import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

from general.mesh_io import read_polygon, write_inp
from sac.sac_env import Sac_Env
from general.smoothing import smooth_mesh
from ebrd.sample_extraction import extract_samples, write_samples
from general.quality import robust_quality
from general.plotting import save_meshes


base_path = Path(__file__).parent.parent
domains_path = base_path / "samples" / "domains"
output_path = base_path / "sac" / "output"
logs_path = output_path / "logs"
eval_path = output_path / "evaluation"
experiments_path = output_path / "experiments"


version = "77"
stage = "0"
method = "sac"


environments = [
    Sac_Env(read_polygon(domains_path / f"{name}.json"))
    for name in ["boundary15", "random1_1", "random2_2"]
]


def prepare_model(method_name, model_path, env):
    loaders = {'a2c': A2C, 'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3}
    return loaders[method_name].load(model_path, env=env)


def rollout(model, env, deterministic=False, render=False):
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        if render:
            env.render()
        if done:
            break
    env.close()
    return info


def evaluation(is_render=False, deterministic=False, indexing=False, save_fig=False, save_samples=False):
    os.makedirs(eval_path / version, exist_ok=True)
    os.makedirs(experiments_path / version, exist_ok=True)

    ckpt = logs_path / method / version / stage / "best_model.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run `python -m sac.train` first.")

    results = {'completed': [], 'n_elements': [], 'n_complete': 0}

    for i, env in enumerate(environments):
        print(f'Evaluating env {i}')

        model = prepare_model(method, ckpt, env)
        info = rollout(model, env, deterministic=deterministic, render=is_render)

        results['completed'].append(info['is_complete'])
        results['n_elements'].append(len(env.mesh.generated_quads))
        results['n_complete'] += 1 if info['is_complete'] else 0

        tag = f"{method}_env_{i}_{'T' if deterministic else 'F'}"

        if info['is_complete']:
            smooth_mesh(env.mesh, env.boundary, env.mesh.vertices())

        if save_fig and env.mesh.generated_quads:
            save_meshes(env, eval_path / version / f"{tag}.png", quads=env.mesh.generated_quads,
                            quality=False, indexing=indexing, style='k-')

            write_inp(env.mesh, eval_path / version / f"{tag}.inp")

            with open(experiments_path / version / f"{tag}_history_info", 'w') as fw:
                json.dump(env.history_info, fw)

            q = [robust_quality(quad) for quad in env.mesh.generated_quads]
            print(f"element quality mean/std: {np.mean(q):.3f} / {np.std(q):.3f}")

        if save_samples and env.mesh.generated_quads:
            samples, output_types, outputs = extract_samples(env.mesh, env.mesh.generated_quads, 2, 3, radius=4)
            write_samples(eval_path / version / f"{tag}.json",
                             {'samples': samples, 'output_types': output_types, 'outputs': outputs})

    with open(eval_path / version / 'evaluation.txt', 'w') as outfile:
        json.dump(results, outfile)
    print(f"Wrote results + meshes + history to {eval_path / version} and {experiments_path / version}")


def replication_evaluation(is_render=False, deterministic=False, save_fig=False):
    os.makedirs(eval_path / version, exist_ok=True)

    ckpt = logs_path / method / version / stage / "best_model.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run `python -m sac.train` first.")

    env = Sac_Env(read_polygon(domains_path / "boundary_fly_r2.json"))

    model = prepare_model(method, ckpt, env)

    times = 10
    results = {'sac': {'completed': [], 'n_elements': [], 'n_complete': 0}}
    for j in range(times):
        info = rollout(model, env, deterministic=deterministic, render=is_render)
        results['sac']['completed'].append(info['is_complete'])
        results['sac']['n_elements'].append(len(env.mesh.generated_quads))
        results['sac']['n_complete'] += 1 if info['is_complete'] else 0

        if save_fig and info['is_complete']:
            smooth_mesh(env.mesh, env.boundary, env.mesh.vertices())

    with open(eval_path / version / "evaluation_repli.txt", 'w') as outfile:
        json.dump(results, outfile)


def element_number_box_plot():
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


def full_mesh(domain="boundary6",
               ckpt=logs_path / method / "78" / "1" / "best_model.zip",
               attempts=60):
    os.makedirs(eval_path, exist_ok=True)

    env = Sac_Env(read_polygon(domains_path / f"{domain}.json"))

    model = prepare_model(method, ckpt, env)

    out = eval_path / f"full_mesh_{domain}.png"

    for k in range(attempts):
        info = rollout(model, env)

        coverage = sum(m.area() for m in env.mesh.generated_quads) / env.mesh.original_area

        if info['is_complete']:
            smooth_mesh(env.mesh, env.boundary, env.mesh.vertices())

            save_meshes(env, out, quads=env.mesh.generated_quads, style='k-')
            print(f"Completed {domain} on attempt {k + 1} "
                  f"({len(env.mesh.generated_quads)} elements, {coverage * 100:.0f}% area). Saved {out}")
            return

    save_meshes(env, out, quads=env.mesh.generated_quads, style='k-')
    print(f"No full completion in {attempts} attempts; saved best-effort partial to {out}")


if __name__ == '__main__':
    full_mesh()
    full_mesh(domain="dolphin1")
    full_mesh(domain="basic")
    full_mesh(domain="basic1")
    # evaluation(is_render=False, deterministic=False, indexing=False, save_fig=True, save_samples=False)
    # replication_evaluation(save_fig=True)
    # element_number_box_plot()

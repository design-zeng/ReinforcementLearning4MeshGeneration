import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import SubplotSpec

from general.tools.mesh_quality_comparison import output_path as comparison_output_path
from ebrd.data_augmentation import (ACTION_TYPES, decode_geometry, compute_quality,
                                    load_samples, augmentation_path)

sns.set_theme(style="darkgrid")


def samples_2_plot_data(samples, quality_threshold=0, include_quality=True):
    data = {
        'Type 0': {
            'Neighbors': {'x': [], 'y': []},
            'RNeighbors': {'x': [], 'y': []},
            'Angles': [],
            'Quality': []
        },
        'Type 1': {
            'Neighbors': {'x': [], 'y': []},
            'RNeighbors': {'x': [], 'y': []},
            'Vertex': {'x': [], 'y': []},
            'Angles': [],
            "Quality": []
        },
        'Type 2': {
            'Neighbors': {'x': [], 'y': []},
            'RNeighbors': {'x': [], 'y': []},
            'Angles': [],
            "Quality": []
        }
    }

    actions = np.asarray(list(map(list.__add__, samples['output_types'], samples['outputs'])))
    observations = np.asarray(samples['samples'])
    for i in range(len(observations)):
        if 'quality' in samples.keys():
            if samples['quality'][i] < quality_threshold:
                continue

        n_points, r_points, rule_type, new_point = decode_geometry(observations[i], actions[i])
        if include_quality:
            element_quality, boundary_quality = compute_quality(observations[i], actions[i])
            quality = math.sqrt(element_quality * boundary_quality)
        else:
            quality = 0

        if rule_type == ACTION_TYPES[0]:
            data['Type 0']['Neighbors']['x'].extend([p.x for p in n_points])
            data['Type 0']['Neighbors']['y'].extend([p.y for p in n_points])
            data['Type 0']['RNeighbors']['x'].extend([p.x for p in r_points])
            data['Type 0']['RNeighbors']['y'].extend([p.y for p in r_points])
            data['Type 0']['Angles'].append(observations[i][-1])
            data['Type 0']['Quality'].append(quality)

        elif rule_type == ACTION_TYPES[2]:
            data['Type 2']['Neighbors']['x'].extend([p.x for p in n_points])
            data['Type 2']['Neighbors']['y'].extend([p.y for p in n_points])
            data['Type 2']['RNeighbors']['x'].extend([p.x for p in r_points])
            data['Type 2']['RNeighbors']['y'].extend([p.y for p in r_points])
            data['Type 2']['Angles'].append(observations[i][-1])
            data['Type 2']['Quality'].append(quality)
        else:
            data['Type 1']['Neighbors']['x'].extend([p.x for p in n_points])
            data['Type 1']['Neighbors']['y'].extend([p.y for p in n_points])
            data['Type 1']['RNeighbors']['x'].extend([p.x for p in r_points])
            data['Type 1']['RNeighbors']['y'].extend([p.y for p in r_points])
            data['Type 1']['Vertex']['x'].append(new_point.x)
            data['Type 1']['Vertex']['y'].append(new_point.y)
            data['Type 1']['Angles'].append(observations[i][-1])
            data['Type 1']['Quality'].append(quality)

    return data


def scatter_plot(samples):
    data = samples_2_plot_data(samples)

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    i = 0
    for k, v in data.items():

        axs[0, i].plot(v['Neighbors']['x'], v['Neighbors']['y'], 'b.',
                    v['RNeighbors']['x'], v['RNeighbors']['y'], 'y.')
        axs[0, i].set_title(k + ": vertex distribution")
        if k == 'Type 1':
            axs[0, i].plot(v['Vertex']['x'], v['Vertex']['y'], 'r.')

        axs[1, i].hist(v['Angles'], 15)
        axs[1, i].set_title(k + ': angle distribution')
        axs[2, i].hist(v['Quality'], 10)
        axs[2, i].set_title(k + ': quality distribution')

        i += 1
    fig.tight_layout()
    plt.show()


def comparison_scatter_plot(samples1, samples2, include_quality=False):
    data1 = samples_2_plot_data(samples1, include_quality=include_quality)
    data2 = samples_2_plot_data(samples2, include_quality=include_quality)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    i = 0
    for k in ["Type 0", "Type 1", "Type 2"]:
        v = data1[k]
        title = k

        axs[0, i].hist(random.sample(v['Angles'], 10000) if len(v['Angles']) > 10000 else v['Angles'], 15)
        axs[0, i].set_title(title, fontsize=18)
        axs[0, i].set_xlim(0.5, 2.5)

        i += 1

    i = 0
    for k in ["Type 0", "Type 1", "Type 2"]:
        v = data2[k]
        title = k

        axs[1, i].hist(random.sample(v['Angles'], 10000) if len(v['Angles']) > 10000 else v['Angles'], 15)
        axs[1, i].set_title(title, fontsize=18)
        axs[1, i].set_xlim(0.5, 2.5)
        i += 1

    grid = plt.GridSpec(2, 3)
    create_subtitle(fig, grid[0, ::], '(a) Samples extracted from Gmsh', fontsize=20)
    create_subtitle(fig, grid[1, ::], '(b) Samples generated by FreeMesh-DG', fontsize=20)

    fig.tight_layout()
    plt.show()


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str, fontsize: int):
    row = fig.add_subplot(grid)
    row.set_title(f'{title}\n', fontweight='semibold', fontsize=fontsize)
    row.set_frame_on(False)
    row.axis('off')


if __name__ == '__main__':
    generated_path = augmentation_path / "run1" / "training_samples.json"
    if not generated_path.exists():
        raise FileNotFoundError(
            f"No generated samples at {generated_path}. Run `python -m ebrd.train` "
            f"(or `python -m ebrd.data_augmentation`) first.")
    generated = load_samples(generated_path)

    extracted_files = sorted((comparison_output_path / "data_augmentation").glob("*.json"))
    if extracted_files:
        extracted = {'samples': [], 'output_types': [], 'outputs': []}
        for f in extracted_files:
            part = load_samples(f)
            for k in extracted:
                extracted[k].extend(part[k])
        comparison_scatter_plot(extracted, generated, include_quality=False)
    else:
        print(f"No extracted samples under {comparison_output_path / 'data_augmentation'}; "
              f"run extract_samples_from_file() in general.tools.mesh_quality_comparison "
              f"to compare against Gmsh meshes. Plotting the generated samples only.")
        scatter_plot(generated)

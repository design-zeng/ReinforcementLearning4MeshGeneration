import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ebrd.data_augmentation import ACTION_TYPES, decode_geometry, compute_quality, load_samples


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
    type_by_action = {ACTION_TYPES[0]: 'Type 0', ACTION_TYPES[2]: 'Type 2'}
    for i in range(len(observations)):
        if 'quality' in samples.keys():
            if samples['quality'][i] < quality_threshold:
                continue

        neighbor_points, radius_points, rule_type, new_point = decode_geometry(observations[i], actions[i])
        if include_quality:
            element_quality, boundary_quality = compute_quality(observations[i], actions[i])
            quality = math.sqrt(element_quality * boundary_quality)

        else:
            quality = 0

        bucket = data[type_by_action.get(rule_type, 'Type 1')]
        bucket['Neighbors']['x'].extend([p.x for p in neighbor_points])
        bucket['Neighbors']['y'].extend([p.y for p in neighbor_points])
        bucket['RNeighbors']['x'].extend([p.x for p in radius_points])
        bucket['RNeighbors']['y'].extend([p.y for p in radius_points])
        bucket['Angles'].append(observations[i][-1])
        bucket['Quality'].append(quality)
        if 'Vertex' in bucket:
            bucket['Vertex']['x'].append(new_point.x)
            bucket['Vertex']['y'].append(new_point.y)

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

        # angle distribution
        axs[1, i].hist(v['Angles'], 15)
        axs[1, i].set_title(k + ': angle distribution')
        # quality distribution
        axs[2, i].hist(v['Quality'], 10)
        axs[2, i].set_title(k + ': quality distribution')

        i += 1
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Inspect a generated dataset:
    #   scatter_plot(load_samples("<path to training_samples.json>"))
    pass

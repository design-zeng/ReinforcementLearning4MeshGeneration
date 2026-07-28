import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

base_path = Path(__file__).parent.parent.parent
# history-info files are produced by sac/infer.py's evaluation()
experiments_path = base_path / "sac" / "output" / "experiments"

def read_meshing_history_info(filename):
    data = None
    with open(filename, 'r+') as fr:
        data = json.load(fr)

    new_data = {'0': [],
                '1': []}
    for k, v in data.items():
        new_data['0' if k == '-1' or k == '1' else '1'].extend(v)

    return new_data


def geometries():
    # Auto-discover the history-info files produced by sac/infer.py evaluation().
    files = sorted(experiments_path.rglob("*_history_info"))
    if not files:
        raise FileNotFoundError(
            f"No *_history_info under {experiments_path}. "
            f"Run `python -m sac.infer` (save_fig=True) first.")
    geometries = {f.name.replace("_history_info", ""): str(f) for f in files}

    data = pd.DataFrame({"Number": [], "Rule type": [], "Domains": []})
    all_data = pd.DataFrame({"Number": [], "Domains": []})
    for k, v in geometries.items():
        info = read_meshing_history_info(v)
        for rule, _v in info.items():

            r = {"Number": [len(_v)], "Rule type": [rule], "Domains": [k]}
            r = pd.DataFrame(r)
            data = pd.concat([data, r])
        _r = {"Number": [sum([len(v) for k, v in info.items()])], "Domains": [k]}
        _r = pd.DataFrame(_r)
        all_data = pd.concat([all_data, _r])

    return data, all_data

def plot_history_info(data_frame):
    # matplotlib.rc_file_defaults()
    # ax1 = sns.set_style(style=None, rc=None)
    # fig, ax1 = plt.subplots(figsize=(12, 6))
    scatter = sns.barplot(x='Domains', y='Number', data=data_frame[0], hue='Rule type')
    i = 0
    for p in scatter.patches:
        scatter.annotate(format(p.get_height(), '.0f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')
    # scatter.legend(title='Geometry domains')
    # ax2 = ax1.twinx()
    # sns.lineplot(data=data_frame[1], x='Domains', y='Number', marker='o', sort=False, hue=None)

    plt.show()


if __name__ == '__main__':
    plot_history_info(geometries())
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sns.set_theme(style="darkgrid")

base_path = Path(__file__).parent.parent.parent
experiments_path = base_path / "sac" / "output" / "experiments"


def read_meshing_history_info(filename):
    with open(filename) as fr:
        rewards_by_rule = json.load(fr)
    counts = {"0": [], "1": []}
    for rule, rewards in rewards_by_rule.items():
        counts["1" if rule == "0" else "0"].extend(rewards)
    return counts


def load_action_counts():
    files = sorted(experiments_path.rglob("*_history_info"))
    if not files:
        raise FileNotFoundError(
            f"No *_history_info under {experiments_path}. "
            f"Run `python -m sac.infer` (save_fig=True) first.")
    rows = []
    for f in files:
        domain = f.name.replace("_history_info", "")
        for action_type, rewards in read_meshing_history_info(f).items():
            rows.append({"Domains": domain, "Action type": action_type, "Number of actions": len(rewards)})
    return pd.DataFrame(rows)


def plot_action_counts(data_frame):
    ax = sns.barplot(x="Domains", y="Number of actions", hue="Action type", data=data_frame)
    for p in ax.patches:
        assert isinstance(p, Rectangle)
        ax.annotate(format(p.get_height(), ".0f"),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 9), textcoords="offset points")
    plt.show()


if __name__ == "__main__":
    plot_action_counts(load_action_counts())

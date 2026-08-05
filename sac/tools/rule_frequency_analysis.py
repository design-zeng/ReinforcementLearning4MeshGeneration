import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


base_path = Path(__file__).parent.parent.parent

# Must match the run evaluated by sac.infer.evaluation().
version = "77"
method = "sac"
experiments_path = base_path / "sac" / "output" / "experiments" / version

# Eval domains in the same order as sac.infer.environments; evaluation() wrote
# one history-info file per domain, named "<method>_env_<i>_F_history_info".
domains = ["boundary15", "random1_1", "random2_2"]


def read_meshing_history_info(filename):
    with open(filename) as f:
        rewards_by_rule = json.load(f)
    counts = {"0": [], "1": []}
    for rule, rewards in rewards_by_rule.items():
        counts["1" if rule == "0" else "0"].extend(rewards)
    return counts


def load_action_counts():
    rows = []
    for i, domain in enumerate(domains):
        history_file = experiments_path / f"{method}_env_{i}_F_history_info"
        if not history_file.exists():
            raise FileNotFoundError(
                f"{history_file} not found. "
                f"Run `python -m sac.infer` (evaluation with save_fig=True) first.")
        for action_type, rewards in read_meshing_history_info(history_file).items():
            rows.append({"Domains": domain, "Action type": action_type, "Number of actions": len(rewards)})
    return pd.DataFrame(rows)


def plot_action_counts(data_frame):
    sns.set_theme(style="darkgrid")
    ax = sns.barplot(x="Domains", y="Number of actions", hue="Action type", data=data_frame)
    for p in ax.patches:
        assert isinstance(p, Rectangle)
        ax.annotate(format(p.get_height(), ".0f"),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 9), textcoords="offset points")
    plt.show()


if __name__ == "__main__":
    plot_action_counts(load_action_counts())

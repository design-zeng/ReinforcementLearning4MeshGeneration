from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


base_path = Path(__file__).parent.parent.parent
logs_path = base_path / "sac" / "output" / "logs"


def load_log(path, tags, name=None, MAX=None):
    if isinstance(path, str):
        runlog_data = pd.DataFrame({"metric": [], "Averaged return": [], "Time step": [], "method": []})
        try:
            event_acc = EventAccumulator(path, {"scalars": 0})
            event_acc.Reload()
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                if MAX:
                    event_list = [e for e in event_list if e.step <= MAX]
                values = [e.value for e in event_list]
                step = [e.step for e in event_list]
                r = {"metric": [tag] * len(step), "Averaged return": values, "Time step": step,
                     "method": [name] * len(step)}
                runlog_data = pd.concat([runlog_data, pd.DataFrame(r)])
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
        return runlog_data
    elif isinstance(path, dict):
        runlog_data = pd.DataFrame({"metric": [], "Averaged return": [], "Time step": [], "method": []})
        for k, v in path.items():
            runlog_data = pd.concat([runlog_data, load_log(v, tags, k, MAX=MAX)])
        return runlog_data
    return pd.DataFrame({"metric": [], "Averaged return": [], "Time step": [], "method": []})


def plot_tensorflow_log(data_frame, title="Run"):
    sns.set_theme(style="darkgrid")
    scatter = sns.lineplot(x='Time step', y='Averaged return', data=data_frame, errorbar=None, hue='method')
    scatter.legend(title=title)
    plt.show()


if __name__ == '__main__':
    tags = ["rollout/ep_rew_mean"]

    event_dirs = sorted({p.parent for p in logs_path.rglob("events.out.tfevents.*")})
    if not event_dirs:
        raise FileNotFoundError(
            f"No TensorBoard event files under {logs_path}. Run `python -m sac.train` first.")
    runs = {str(d.relative_to(logs_path)): str(d) for d in event_dirs}

    log_data = load_log(runs, tags, MAX=1200000)
    plot_tensorflow_log(log_data, "Action radius")
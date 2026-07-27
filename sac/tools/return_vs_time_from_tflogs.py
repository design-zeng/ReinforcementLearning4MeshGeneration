import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
import pandas as pd
import traceback
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
sns.set_theme(style="darkgrid")

base_path = Path(__file__).parent.parent.parent
config = configparser.ConfigParser()
config.read(f'{base_path}/config')


def load_log(path, tags, name=None, MAX=None):
    if isinstance(path, str):
        DEFAULT_SIZE_GUIDANCE = {
            'compressedHistograms': 1,
            'images': 1,
            'scalars': 0,
            'histograms': 1
        }
        runlog_data = pd.DataFrame({"metric": [], "Averaged return": [], "Time step": [], "method": []})
        try:
            event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
            event_acc.Reload()
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                if MAX:
                    N = 0
                    while True:
                        if step[N] > MAX:
                            break
                        N += 1
                    values = values[: N]
                    step = step[:N]
                r = {"metric": [tag] * len(step), "Averaged return": values, "Time step": step,
                     "method": [name] * len(step) if name is not None else []}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
        # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()
        return runlog_data
    elif isinstance(path, dict):
        runlog_data = pd.DataFrame({"metric": [], "Averaged return": [], "Time step": [], "method": []})
        for k, v in path.items():
            _res = load_log(v, tags, k, MAX=MAX)
            runlog_data = pd.concat([runlog_data, _res])
        return runlog_data

def plot_tensorflow_log(data_frame):

    # Loading too much data is slow...
    # sns.set(font_scale=1.1)
    scatter = sns.lineplot(x='Time step', y='Averaged return', data=data_frame, ci=None, hue='method')
    # scatter.set_xticklabels(scatter.get_xmajorticklabels(), fontsize = 16)
    # scatter.set_yticklabels(scatter.get_ymajorticklabels(), fontsize = 16)
    # scatter.legend(title="Observation range")
    # scatter.legend(title='RL methods')
    # scatter.legend(title='Training domains')
    # scatter.legend(title='NN structures')
    # scatter.legend(title='Random seeds')
    scatter.legend(title="Action radius")
    plt.show()

    # nn structure comparison
    # fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
    # # fig.suptitle('1 row x 2 columns axes with no data')
    # axes[0].set_title('(a) Training domain')
    # axes[0].grid(False)
    # axes[0].set_frame_on(False)
    # axes[0].set_xticks([])
    # axes[0].set_yticks([])
    # img = mpimg.imread("C:\\Users\\umroot\OneDrive - Concordia University - Canada\\research\PAMI\pics\\training domain.png")
    # axes[0].imshow(img)
    #
    # scatter = sns.lineplot(ax=axes[1], x='Time step', y='Averaged return', data=data_frame, ci=None, hue='method')
    # scatter.legend(title="NN structures")
    # axes[1].set_title('(b) Neural network structure comparison')
    # plt.show()


if __name__ == '__main__':
    # log_file = config['default']['log'] + '\ppo_tensorboard\\22\PPO_1\\events.out.tfevents.1617035015.Krakta.21728.0'

    tags = [
        # "eval/mean_ep_length",
        # "eval/mean_reward",
        # "rollout/ep_len_mean",
        "rollout/ep_rew_mean",
        # "time/fps",
        # "train/approx_kl",
        # "train/clip_fraction",
        # "train/clip_range",
        # "train/entropy_loss",
        # "train/explained_variance",
        # "train/learning_rate",
        # "train/loss",
        # "train/policy_gradient_loss",
        # "train/std",
        # "train/value_loss"
    ]

    # method_logs = {
    #     # 'a2c': config['default']['log'] + '\\a2c_tensorboard\\5\A2C_1\events.out.tfevents.1617247183.Krakta.12680.0',  # 'D:/meshingData/baselines/logs/a2c_mesh/3/best_model'
    #     'PPO': config['default']['log'] + '\\ppo_tensorboard\\64\curriculum\\0\PPO_1\\events.out.tfevents.1641156620.fniss.6656.0',  # 16/3541; 21/3828; 22/4077, 4170, 4624
    #     'DDPG': config['default']['log'] + '\\ddpg_tensorboard\\6\curriculum\\0\DDPG_1\\events.out.tfevents.1641411894.fniss.3568.0',
    #     'TD3': config['default']['log'] + '\\td3_tensorboard\\6\curriculum\\0\TD3_1\\events.out.tfevents.1641416743.fniss.18848.0',
    #     'SAC': config['default']['log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',
    # }

    # NN_logs = {
    # # 'D:/meshingData/baselines/logs/a2c_mesh/3/best_model'
    #     'S1': config['default'][
    #               'log'] + '\\sac_tensorboard\\34\curriculum\\0\SAC_1\\events.out.tfevents.1626276119.Krakta.14496.0',
    #     'S2': config['default']['log'] + '\sac_tensorboard\\52\curriculum\\0\SAC_1\\events.out.tfevents.1628696043.Krakta.8800.0',
    # # 16/3541; 21/3828; 22/4077, 4170, 4624
    #     'S3': config['default']['log'] + '\\sac_tensorboard\\36\curriculum\\0\SAC_1\\events.out.tfevents.1626441362.Krakta.20180.0',
    #     'S4': config['default'][
    #               'log'] + '\sac_tensorboard\\38\curriculum\\0\SAC_1\\events.out.tfevents.1626541790.Krakta.12388.0',
    # }

    # State_logs = {
    #     # 'D:/meshingData/baselines/logs/a2c_mesh/3/best_model'
    #     'O1: 4_2_3': config['default'][
    #                'log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',
    #     # 'state_4_3_3': config['default'][
    #     #           'log'] + '\sac_tensorboard\\39\curriculum\\0\SAC_1\\events.out.tfevents.1626578162.Krakta.20832.0',
    #     # # 16/3541; 21/3828; 22/4077, 4170, 4624
    #     # 'state_4_2_4': config['default'][
    #     #           'log'] + '\\sac_tensorboard\\40\curriculum\\0\SAC_1\\events.out.tfevents.1626618291.Krakta.4880.0',
    #     'O2: 6_2_3': config['default']['log'] +
    #                    '\\sac_tensorboard\\72\curriculum\\0\SAC_1\\events.out.tfevents.1641604788.frakta.11592.0',
    #     'O3: 6_3_3': config['default']['log'] +
    #                  '\\sac_tensorboard\\74\curriculum\\0\SAC_1\\events.out.tfevents.1641692276.frakta.2884.0',
    #     'O4: 6_3_4': config['default']['log'] +
    #                    '\\sac_tensorboard\\73\curriculum\\0\SAC_1\\events.out.tfevents.1641604867.frakta.18084.0',
    # }

    # training_domain_logs = {
    #     'T1': config['default'][
    #                'log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',
    #     'T2': config['default'][
    #                'log'] + '\\sac_tensorboard\\65\curriculum\\0\SAC_1\\events.out.tfevents.1641445982.fniss.20348.0',
    #     'T3': config['default'][
    #                'log'] + '\\sac_tensorboard\\66\curriculum\\0\SAC_1\\events.out.tfevents.1641495265.fniss.24084.0',
    # }
    #
    NN_logs = {
    # 'D:/meshingData/baselines/logs/a2c_mesh/3/best_model'

        'S1': config['default']['log'] + '\sac_tensorboard\\69\curriculum\\0\SAC_1\\events.out.tfevents.1641529497.frakta.11340.0',
        'S2': config['default'][
                  'log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',

        'S3': config['default']['log'] + '\\sac_tensorboard\\71\curriculum\\0\SAC_1\\events.out.tfevents.1641532084.fniss.13632.0',

        'S4': config['default']['log'] + '\\sac_tensorboard\\36\curriculum\\0\SAC_1\\events.out.tfevents.1626441362.Krakta.20180.0',

        # 'S4': config['default'][
        #           'log'] + '\sac_tensorboard\\70\curriculum\\0\SAC_1\\events.out.tfevents.1641529743.fniss.14712.0',
    }
    #
    # compare_a2c = {
    #     'A2C': config['default']['log'] + '\\a2c_tensorboard\\6\curriculum\\0\A2C_1\\events.out.tfevents.1641587950.fniss.8060.0',
    #     'SAC': config['default'][
    #                'log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',
    # }
    #
    # seed_logs = {
    #     'Seed A': config['default'][
    #                'log'] + '\\sac_tensorboard\\67\curriculum\\0\SAC_1\\events.out.tfevents.1641482397.frakta.848.0',
    #     'Seed B': config['default'][
    #                'log'] + '\\sac_tensorboard\\68\curriculum\\0\SAC_1\\events.out.tfevents.1641495315.frakta.12756.0',
    #     'Seed C': config['default'][
    #                'log'] + '\\sac_tensorboard\\63\curriculum\\0\SAC_1\\events.out.tfevents.1641155959.fniss.15168.0',
    # }

    # radius_logs = {
    #     'R1': config['default'][
    #               'log'] + '\\sac_tensorboard\\76\curriculum\\0\SAC_1\\events.out.tfevents.1662335075.Jay.20700.0',
    #     # 'R2': config['default']['log'] +
    #     #                '\\sac_tensorboard\\72\curriculum\\0\SAC_1\\events.out.tfevents.1641604788.frakta.11592.0',
    #     'R2': config['default'][
    #                   'log'] + '\\sac_tensorboard\\sac_75\curriculum\\0\SAC_1\\events.out.tfevents.1662188104.Jay.22376.0',
    #
    #     'R3': config['default'][
    #               'log'] + '\\sac_tensorboard\\77\curriculum\\0\SAC_1\\events.out.tfevents.1662358240.Jay.26836.0'
    #
    # }

    log_data = load_log(radius_logs, tags, MAX=1200000)
    plot_tensorflow_log(log_data)
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set_theme(style="darkgrid")


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
    geometries = {
        'Domain 4': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1141_env_0_history_info',
        'Domain 5': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1141_env_2_history_info',
        'Domain 6': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1141_env_4_history_info',
        # 'Domain 10': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1141_env_1_history_info',
        # 'Domain 11': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1141_env_3_history_info',
        'Domain 7': 'D:\meshingData\\baselines\logs\experiments\sac_72_2 - Copy\\sac_0_1167_env_5_history_info',
    }

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
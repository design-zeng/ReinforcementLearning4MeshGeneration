import os
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3

from general.polygon_generators import read_polygon
from general.boundary_env import BoudaryEnv

# sns.set_theme(style="darkgrid")

base_path = Path(__file__).parent.parent
domains_path = base_path / "domains"
output_path = base_path / "sac" / "output"
logs_path = output_path / "logs"                 # trained checkpoints written by train.py: logs/<method>/<version>/<stage>/<checkpoint>
eval_path = output_path / "evaluation"           # evaluation figures, samples, result files
experiments_path = output_path / "experiments"   # history-info dumps
pics_path = output_path / "pics"                 # density images for element_number_box_plot


version = 'sac_training_domain_compare_66' # sac_4_ann
total_timesteps = 4000000
seed=999
learning_rate=1e-4

def prepare_eval_envs():
    domains = [
        domains_path / "boundary15.json",
        domains_path / "random1_1.json",  # medium
        domains_path / "random2_2.json",  # hard
    ]
    domains = [BoudaryEnv(read_polygon(d), experiment_version=version, env_name=i) for i, d in enumerate(domains)]
    # for d in domains:
    #     d.estimate_area_range()
    return domains

def prepare_model(method_name, model_path, env):

    # env = Monitor(o_env, logs_path / "a2c" / version)
    if method_name == 'a2c':
        # the policy_kwargs are automatically loaded
        model = A2C.load(model_path, env=env)

    elif method_name == 'ddpg':
        # the policy_kwargs are automatically loaded
        model = DDPG.load(model_path, env=env)

    elif method_name == 'ppo':
        model = PPO.load(model_path, env=env)

    elif method_name == 'sac':
        model = SAC.load(model_path, env=env)

    elif method_name == 'td3':
        model = TD3.load(model_path, env=env)

    return model


def plot_elements_area(elements):
    fig, ax = plt.subplots()
    area = sorted([e.compute_area()[0] for e in elements])
    _min, _max = min(area), max(area)
    ax.bar([], area, label='Area')

    plt.show()


def evaluation(is_render=False, deterministic=False, indexing=False, save_fig=False, save_samples=False):
    os.makedirs(eval_path / version, exist_ok=True)
    envs = prepare_eval_envs()
    is_random = 'T' if deterministic else 'F'
    # Checkpoints are loaded from train.py's layout: logs/<method>/<model_version>/<stage>/<checkpoint>
    model_version = "66"
    stage = "0"
    START = 1200 #3348
    END = 1210
    # agents = [529, 554, 621, 778, 800, 838, 881, 897, 907, 945, 950, 953, 1010, 1080, 1106, 1124, 1189, 1198,
    #           1349, 1402, 1470, 1598, 1600]

    # agents = [1141] # sac_72, [979, 982, 1141, 1167]
    # agents = [1108] # 1613 (31)

    replication = {
        'times': 1
    }

    methods = {
        'a2c': [], # logs_path / "a2c" / "3" / stage / "best_model"
        # 'ppo': [logs_path / "ppo" / "38" / stage / "2090"], #16/3541; 21/3828; 22/4077, 4170, 4624; 31/4103
        # 'ppo': [logs_path / "ppo" / model_version / stage / str(i) for i in range(START, END)], #
        # 'ppo': [logs_path / "ppo" / model_version / stage / "best_model"],
        # 'ddpg': [logs_path / "ddpg" / model_version / stage / str(i) for i in range(START, END)],
        'sac': [logs_path / "sac" / model_version / stage / str(i) for i in range(START, END)], # agents
        # 'sac': [logs_path / "sac" / model_version / stage / str(i) for i in range(START, END)],  # range(START, END)
        # 'ppo': [logs_path / "ppo" / "43" / stage / "mesh"],
        # 'td3': [logs_path / "td3" / model_version / stage / str(i) for i in range(START, END)]
    }
    results = {
        'a2c': [],  # logs_path / "a2c" / "3" / stage / "best_model"
        # 'ppo': [logs_path / "ppo" / "38" / stage / "1893"], #16/3541; 21/3828; 22/4077, 4170, 4624; 31/4103
        # 'ppo': {v.name: {'completed': [],
        #         'n_elements': [],
        #         'n_complete': 0} for v in methods['ppo']},
        # 'ddpg': {v.name: {'completed': [],
        #         'n_elements': [],
        #         'n_complete': 0} for v in methods['ddpg']},
        'sac': {v.name: {'completed': [],
                        'n_elements': [],
                        'n_complete': 0} for v in methods['sac']},
        # 'td3': {v.name: {'completed': [],
        #         'n_elements': [],
        #         'n_complete': 0} for v in methods['td3']},
    }

    time_costs = []
    for k in methods.keys():
        for v in methods[k]:
            for i, env in enumerate(envs):
                # if i < 3:
                #     continue
                print(f'Starting for model {v} with env {i}')
                model = prepare_model(k, v, env)

                # for j in range(replication['times']):
                obs = env.reset()
                # env.boundary.savefig(eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_00.png", dpi=200, style='b.-')
                # print(len(env.original_vertices), env.boundary.get_perimeter())
                start = time.time()
                while True:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    obs, rewards, dones, info = env.step(action)
                    if is_render:
                        env.render()
                    if dones:
                        break

                # time_costs.append(time.time() - start)
                # if len(time_costs) == 10:
                #     print(np.mean(time_costs), np.std(time_costs))
                #     break

                print(f'Meshing running time: {time.time() - start}s')
                env.close()
                results[k][v.name]['completed'].append(info['is_complete'])
                results[k][v.name]['n_elements'].append(len(env.generated_meshes))
                results[k][v.name]['n_complete'] += 1 if info['is_complete'] else 0
                # plot_elements_area(env.generated_meshes)
                # env.smooth(env.boundary.vertices)

                if save_fig:
                    if info['is_complete']:
                        env.save_meshes(eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_{is_random}.png",
                                        meshes=env.generated_meshes, quality=False, type=4,
                                        indexing=indexing, style='k-')
                        print(np.mean(
                            [env.get_quality(env.generated_meshes[i], 4) for i in range(len(env.generated_meshes))]))
                        print(np.std(
                            [env.get_quality(env.generated_meshes[i], 4) for i in range(len(env.generated_meshes))]))

                        # env.smooth(env.boundary.vertices)
                        # env.save_meshes(eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_{is_random}_smoothed.png", meshes=env.generated_meshes,
                        #                         indexing=indexing, style='k-')
                        # pass

                        # env.save_history_info(experiments_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_history_info")

                        # env.write_generated_elements_2_file(
                        #     eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_{is_random}.inp")
                    else:
                        env.save_meshes(eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_{is_random}.png",
                                        meshes=env.generated_meshes, quality=False, type=4,
                                        indexing=indexing, style='k-')
                        # pass
                        print(np.mean([env.get_quality(env.generated_meshes[i], 4) for i in range(len(env.generated_meshes))]))
                        print(np.std([env.get_quality(env.generated_meshes[i], 4) for i in range(len(env.generated_meshes))]))

                        # break
                if save_samples:
                    if len(env.generated_meshes):
                        samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4)
                        env.save_samples(eval_path / version / f"{k}_{v.parent.name}_{v.name}_env_{i}_{is_random}.json",
                                         {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
                        print("Saved!")

    with open(eval_path / version / 'evaluation.txt', 'w') as outfile:
        json.dump(results, outfile)


def replication_evaluation(is_render=False, deterministic=False, indexing=False, save_fig=False, save_samples=False):
    os.makedirs(eval_path / version, exist_ok=True)
    model_path = logs_path / "sac" / "34" / "0" / "889"
    env = BoudaryEnv(read_polygon(domains_path / "boundary_fly_r2.json"))
    is_random = 'T' if deterministic else 'F'
    model = prepare_model('sac', model_path, env)
    times = 10

    results = {
        'sac': {
            'completed': [],
            'n_elements': [],
            'n_complete': 0}
    }

    for j in range(times):
        obs = env.reset()
        # env.boundary.savefig(eval_path / version / f"{model_path.parent.name}_{model_path.name}_env_00.png", style='b.-')
        # print(len(env.original_vertices), env.boundary.get_perimeter())
        # start = time.time()
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(action)
            if is_render:
                env.render()
            if dones:
                break
        # print(f'Meshing running time: {time.time() - start}s')
        env.close()
        results['sac']['completed'].append(info['is_complete'])
        results['sac']['n_elements'].append(len(env.generated_meshes))
        results['sac']['n_complete'] += 1 if info['is_complete'] else 0
        # plot_elements_area(env.generated_meshes)
        # env.smooth(env.boundary.vertices)

        if save_fig:
            if info['is_complete']:
                # env.save_meshes(eval_path / version / f"sac_{model_path.parent.name}_{model_path.name}_env_{j}_{is_random}.png",
                #                 meshes=env.generated_meshes, quality=True, type=4,
                #                 indexing=indexing, style='k-')
                env.smooth(env.boundary.vertices)
                # env.save_meshes(eval_path / version / f"{model_path.parent.name}_{model_path.name}_{j}_{is_random}_smoothed.png",
                #                 meshes=env.generated_meshes, indexing=indexing, style='k-')

                # env.write_generated_elements_2_file(
                #     eval_path / version / f"sac_{model_path.parent.name}_{model_path.name}_env_{j}_{is_random}.inp")
            # else:
            #     env.save_meshes(
            #         eval_path / version / f"{model_path.parent.name}_{model_path.name}_{j}_{is_random}_smoothed.png",
            #         meshes=env.generated_meshes, indexing=indexing, style='k-')
            #     # break
        if save_samples:
            if len(env.generated_meshes):
                samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4)
                env.save_samples(eval_path / version / f"{model_path.parent.name}_{model_path.name}_{is_random}.json",
                                 {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
                print("Saved!")

    with open(eval_path / version / "evaluation_repli.txt", 'w') as outfile:
        json.dump(results, outfile)


def element_number_box_plot():
    data = pd.DataFrame({"Element number": [], "model": []})

    #sac/64/473
    sparse = {"model": ['Sparse\ndensity'] * 10, "Element number": [69, 92, 72, 92, 84, 77, 78, 74, 85, 80]}
    sparse = pd.DataFrame(sparse)
    data = pd.concat([data, sparse])
    # sac/56/886
    medium = {"model": ['Medium\ndensity'] * 10, "Element number": [100, 95, 95, 108, 94, 103, 95, 97, 100, 100]}
    medium = pd.DataFrame(medium)
    data = pd.concat([data, medium])
    # sac/60/106
    dense = {"model": ['Dense\ndensity'] * 10, "Element number": [165, 128, 170, 187, 156, 131, 146, 167, 136, 138]}
    dense = pd.DataFrame(dense)
    data = pd.concat([data, dense])

    # fig, axes = plt.subplots(2, 1)
    fig = plt.figure(figsize=(6, 6))
    # # fig.suptitle('1 row x 2 columns axes with no data')
    # axes[0].set_title('(a) Training domain')
    grid = plt.GridSpec(2, 2)
    img_p = fig.add_subplot(grid[:1, :1])
    img_p.grid(False)
    img_p.set_frame_on(False)
    img_p.set_xticks([])
    img_p.set_yticks([])
    img_p.set_title('(a) Sparse density')
    img = mpimg.imread(pics_path / "sparse.png")
    img_p.imshow(img)

    img_p1 = fig.add_subplot(grid[:1, 1:])
    img_p1.grid(False)
    img_p1.set_frame_on(False)
    img_p1.set_xticks([])
    img_p1.set_yticks([])
    img_p1.set_title('(b) Medium density')
    img1 = mpimg.imread(pics_path / "medium.png")
    img_p1.imshow(img1)

    img_p2 = fig.add_subplot(grid[1:, :1])
    img_p2.grid(False)
    img_p2.set_frame_on(False)
    img_p2.set_xticks([])
    img_p2.set_yticks([])
    img_p2.set_title('(c) Dense density')
    img2 = mpimg.imread(pics_path / "dense.png")
    img_p2.imshow(img2)

    # scatter = sns.lineplot(ax=axes[1], x='Time step', y='Average return', data=data_frame, ci=None, hue='method')
    # scatter.legend(title="NN structures")
    # axes[1].set_title('(b) Neural network structure comparison')
    # plt.show()
    #ax = axes[1],
    box = fig.add_subplot(grid[1:, 1:])
    ax = sns.boxplot(ax=box, x="model", y="Element number", data=data,) #hue='model'
    # ax.legend(title=None)
    ax.set_title('(d) Element numbers for each density')
    ax.xaxis.set_label_text('foo')
    ax.xaxis.label.set_visible(False)
    fig.tight_layout()
    # ax.margins(1, 0)
    plt.show()

if __name__ == '__main__':
    os.makedirs(eval_path / version, exist_ok=True)
    evaluation(is_render=False, deterministic=False, indexing=False, save_fig=True, save_samples=False)
    # replication_evaluation(is_render=False, deterministic=False, indexing=False, save_fig=True, save_samples=False)
    # element_number_box_plot()

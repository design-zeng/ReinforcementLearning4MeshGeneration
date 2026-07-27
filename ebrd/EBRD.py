import json
import configparser
import os
import time
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from general.polygon_generators import read_polygon
from general.boundary_env import BoudaryEnv
from ebrd.data_augmentation import MeshAugmentation, sampling_main

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/')

base_path = Path(__file__).parent.parent
config = configparser.ConfigParser()
config.read(f'{base_path}/config')

base_path = 'D:\\meshingData\\ANN\\'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = f"{base_path}models/ea_t4_2/ebrd_model_1.pt"
# version = 'ea_training_28'

# os.makedirs(f'{base_path}models/{version}/', exist_ok=True)
# os.makedirs(f'{base_path}plots/{version}/', exist_ok=True)
# os.makedirs(f'{base_path}samples/{version}/', exist_ok=True)
# os.makedirs(f'{base_path}elements/{version}/', exist_ok=True)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 18
        self.action_space = 2
        self.type_space = 3

        self.fc1 = nn.Linear(self.state_space, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.action_head = nn.Linear(16, self.action_space)
        self.type_head = nn.Linear(16, self.type_space)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        x5 = F.relu(self.fc5(x4))
        action = self.action_head(x5)
        one_hot_types = self.type_head(x5)

        return action, one_hot_types

def get_action(state, model):
    state = torch.FloatTensor(state).to(device)
    action, type_values = model(state)
    return action.tolist(), float(torch.argmax(type_values) / 2)

def load_training_data(file_name):
    with open(file_name, 'r') as fr:
        data = json.loads(fr.read())
    return data

def build_training_data(data):
    inputs = np.array(data['samples'])
    outputs = np.array(data['outputs'])
    output_types = np.array(data['output_types'])

    x = torch.from_numpy(inputs).float().to(device)
    y = torch.from_numpy(np.concatenate((output_types, outputs), axis=1)).float().to(device)
    return x, y

def training_model(model, x, y, model_path, tensorboard_log):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 3e-4
    epoches = 500000

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(tensorboard_log)
    running_loss = []

    for t in range(epoches):
        y_actions, y_types = model(x)
        y_pred = torch.cat([y_types, y_actions], 1).to(device)

        loss = loss_fn(y_pred, y)
        if loss < 0.01:
            break
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        if t % 1000:
            writer.add_scalar('training loss', sum(running_loss[-1000:]) / 1000, t)

    torch.save(model.state_dict(), model_path)

def map_type_to_tensors(tensor):
    result = []
    for value in tensor:
        if value.item() == 0:
            result.append(0)
        elif value.item() == 0.5:
            result.append(1)
        elif value.item() == 1:
            result.append(2)
        else:
            raise ValueError("Unsupported value: {}".format(value.item()))
    return torch.tensor(result)

def train_ch3(train_data, model, num_epoches, batch_size, tensorboard_log, model_path, lr=None):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    classification_loss_fn = nn.CrossEntropyLoss(reduction='sum')  # Cross Entropy Loss for one-hot
    regression_loss_fn = nn.MSELoss(reduction='sum')  # Mean Squared Error Loss for continuous output

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(tensorboard_log)

    for t in range(num_epoches):
        size = len(train_dataloader)
        sum_loss = 0
        for batch, (x, y) in enumerate(train_dataloader):
            y_actions, y_types = model(x)

            classification_loss = classification_loss_fn(y_types, map_type_to_tensors(y[:, 0]).to(device))
            regression_loss = regression_loss_fn(y_actions, y[:, 1:].to(device))

            loss = classification_loss + regression_loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            sum_loss += loss.item()
            if batch % 10:
                writer.add_scalar('training loss',
                                  sum_loss / 1000,
                                  t * size + batch)

        print(f"loss: {sum_loss / size:>7f}  [{t:>5d}/{num_epoches:>5d}]")

        # if t % 2000 == 0:
        #     torch.save(model.state_dict(), f'model_{t/2000}.pt')

    print("Done!")
    torch.save(model.state_dict(), model_path)

def start_training(model_path, data_path, tensorboard_log):
    # x, y = build_training_data(load_training_data(
    #     "D:\meshingData\\baselines\logs\evaluation\sac_4_ann\\sac_0_890_env_0_F.json"))
    x, y = build_training_data(load_training_data(data_path))

    # x, y = build_training_data(load_training_data(
    #     f"{base_path}\samples\ea_bp_t5_2\\ebrd_0.json"))

    # x, y = build_training_data(load_training_data(
    #     f"{base_path}\\models\\ebrd_.json"))

    model = Policy().to(device)

    # training_model(model, x, y, model_path, tensorboard_log)

    train_ch3(list(zip(x, y)), model, 2000, 128, tensorboard_log, model_path)

def training(env, version):
    max_steps = 8000
    episodes = 100
    running_reward = 10
    step = 0
    for i_episode in range(episodes):
        state, ep_reward = env.reset(static=True), 0

        start = time.time()

        for i in range(max_steps):
            step += 1

            action, type_values = get_action(state)

            state, reward, done, _ = env.move(action, round(type_values, 2), 0.9, 0.5)

            # fig = plt.figure(figsize=(15,5))
            #
            # ax = fig.add_subplot(1, 3, 1)
            # [seg.show() for seg in env.boundary.all_segments()]
            # env.boundary.show()
            # ax.set_title("(a) Original boundary")
            #
            # ax1 = fig.add_subplot(1,3,2)
            # [seg.show() for seg in env.boundary.all_segments()]
            # ax1.set_title("(b) Boundary with a generated element")
            # ax2 = fig.add_subplot(1,3,3)
            # [seg.show() for seg in env.updated_boundary.all_segments()]
            # ax2.set_title("(c) Updated boundary")
            # [a.get_xaxis().set_visible(False) for a in fig.axes]
            # [a.get_yaxis().set_visible(False) for a in fig.axes]
            # fig.savefig("teststst.png")

            print(i, reward, len(env.updated_boundary.vertices))
            ep_reward += reward
            if done:
                break

        print(f"Execution time: {time.time() - start}s.")

        if len(env.updated_boundary.vertices) <= 5:
            env.smooth(env.boundary.vertices)
        else:
            env.smooth_pave(env.boundary.vertices, env.updated_boundary.vertices, iteration=400, interior=True)

        # env.plot_meshes(env.generated_meshes, quality=True, type=1)

        # env.boundary.show()
        # env.write_generated_elements_2_file(f"{base_path}elements/{version}/elements_{i_episode}")
        # env.save_meshes(env.generated_meshes[:9], quality=True)
        # env.plot_meshes(env.generated_meshes, quality=True, type=5)
        # env.boundary.show()

        # env.save_meshes(f"{base_path}plots/{version}/{i_episode}.png", env.generated_meshes, quality=True,
        #         #                 indexing=True,
        #         #                 type=1, dpi=300)
        env.boundary.savefig(f"{base_path}plots/{version}/{i_episode}.png", style='k-', dpi=300)
        print("Figure saved!")

        # global model_path
        # model_path = f'{base_path}models/{version}/ebrd_model_{i_episode}.pt'

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # if ep_reward > 100:
        samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4,
                                                               quality_threshold=0.7)
        env.save_samples(f'{base_path}samples/{version}/ebrd_{i_episode}.json',
                         {'samples': samples, 'output_types': output_types, 'outputs': outputs},
                         _type=2)
        # x, y = build_training_data({'samples': [env.points_as_array(sample) for sample in samples],
        #                             'output_types': output_types,
        #                             'outputs': [env.points_as_array(sample) for sample in outputs]})
        x, y = build_training_data({'samples': samples,
                                    'output_types': output_types,
                                    'outputs': outputs})

        # training_model(model, x, y, f'{base_path}models/{version}/ebrd_model_{i_episode}.pt')
        # train_ch3(list(zip(x, y)), model, 10000, 1024)

        print("%d: done %d games, running reward %.3f" % (
            step, i_episode, running_reward,
        ))

        # if running_reward > 2000:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, step))
        #     break

def single_run(env, step, lr_1, lr_2, i_episode):
    state, ep_reward = env.reset(), 0
    for i in range(max_steps):
        step += 1
        # state = state[
        action, type_values = get_action(state)
        state, reward, done, _ = env.move(state, action, round(type_values, 1), lr_1, lr_2)
        # env.boundary.show()
        # print(state, reward, len(env.updated_boundary.vertices))
        ep_reward += reward
        if done:
            break

    # env.boundary.show()
    # env.smooth(env.boundary.vertices, lr_1, lr_2)
    # env.plot_meshes(env.generated_meshes, quality=True)
    # env.boundary.show()
    env.write_generated_elements_2_file(f"{base_path}elements/{version}/elements_{i_episode}")
    env.boundary.savefig(f"{base_path}plots/{version}/{i_episode}.png", f"lr_1: {lr_1}, lr_2: {lr_2}", title="", style="k.-")

    print(f"Thread {i_episode} is finishing! *****************************************")

    global model_path
    model_path = f'{base_path}models/{version}/ebrd_model_{i_episode}.pt'

    # print("%d: done %d games, running reward %.3f" % (
    #     step, i_episode, ep_reward,
    # ))

def prepare_eval_envs():
    domains = []
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/engeer.json')
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/star1.json') #hard
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/random1_1.json') # medium
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/tool2.json')  # medium
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/test2.json')
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/test3.json')
    domains = [BoudaryEnv(read_polygon(d)) for d in domains]
    # for d in domains:
    #     d.estimate_area_range()
    return domains

def data_sampling(data_path, n, threshold):
    # mg = MeshAugmentation([], [])
    # mg.sampling(n=n, threshold=threshold,
    #             file_name=data_path)
    start_time = time.time()
    sampling_main(10, n, threshold,
                  file_name=data_path)
    print('Sampling completed in', time.time() - start_time, 's!')


def evaluation(model_path, version, is_render=False, indexing=False, save_fig=False, save_samples=False):
    os.makedirs(f"{config['default']['augmentation']}/{version}/", exist_ok=True)

    envs = prepare_eval_envs()
    model = Policy().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, env in enumerate(envs):
        # if i < 16:
        #     continue
        print(f'Starting for model with env {i}')

        state = env.reset(static=True)

        # env.boundary.savefig(f"{config['default']['evaluation']}/{version}/ \
        #                             {k}_{v.split('/')[-2]}_{v.split('/')[-1]}_env_00.png", style='b.-')
        # print(len(env.original_vertices), env.boundary.get_perimeter())

        # start = time.time()
        while True:
            action, type_values = get_action(state, model)
            state, reward, done, info = env.move(action, round(type_values, 2))
            if is_render:
                env.render()
            if done:
                break
        # print(f'Meshing running time: {time.time() - start}s')

        env.close()

        # results[k][v.split('/')[-1]]['completed'].append(info['is_complete'])
        # results[k][v.split('/')[-1]]['n_elements'].append(len(env.generated_meshes))
        # results[k][v.split('/')[-1]]['n_complete'] += 1 if info['is_complete'] else 0
        # plot_elements_area(env.generated_meshes)
        # env.smooth(env.boundary.vertices)

        if save_fig:
            if info['is_complete']:
                # env.save_meshes(f"{config['default']['evaluation']}/{version}/{k}_{v.split('/')[-2]}_{v.split('/')[-1]}_env_{i}_{is_random}.png",
                #                 meshes=env.generated_meshes, quality=True, type=4,
                #                 indexing=indexing, style='k-')

                # env.save_meshes(f"{config['default']['augmentation']}/{version}/ebrd_env_{i}.png",
                #                 meshes=env.generated_meshes,
                #                 indexing=indexing, style='k-')
                env.smooth(env.boundary.vertices)
                env.save_meshes(f"{config['default']['augmentation']}/{version}/ebrd_env_{i}__smoothed.png",
                                meshes=env.generated_meshes,
                                        indexing=indexing, style='k-')

                # env.write_generated_elements_2_file(
                #     f"{config['default']['augmentation']}/{version}/ebrd_env_{i}.inp")
            else:
                env.save_meshes(f"{config['default']['augmentation']}/{version}/ebrd_env_{i}.png",
                                meshes=env.generated_meshes, #quality=True, type=4,
                                indexing=indexing, style='k-')
        if save_samples:
            if len(env.generated_meshes):
                samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4)
                env.save_samples(f"{config['default']['augmentation']}/{version}/ebrd_env_{i}.json",
                                 {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
                print("Saved!")


def hyperparameter_search():
    quality_t = [i/100 for i in range(60, 90, 2)]
    for q in quality_t:
        version = f'1_2k_{q}'

        data_path = f"{config['default']['augmentation']}\\1103\\training_samples_{version}.json"
        data_sampling(data_path, n=40000, threshold=q)

        # Meshing model training
        start_time = time.time()
        start_training(f"{config['default']['augmentation']}/{version}.pt",
                       data_path,
                       tensorboard_log=f"{config['default']['augmentation']}/log/{version}/")
        print('Complete training in:', time.time() - start_time, 's.')

        model_path = f"{config['default']['augmentation']}/{version}.pt"
        evaluation(model_path, version, is_render=False, indexing=True,
                   save_fig=True, save_samples=False)


if __name__ == '__main__':
    # for i in range(71, 80):
    #     version = f'data_aug_0.{i/100}'
    #     # initial training
    #     start_training(f"{config['default']['augmentation']}/{version}.pt",
    #                    f"{config['default']['augmentation']}\\1\\training_samples_{i/100}.json",
    #                    tensorboard_log=f"{config['default']['augmentation']}/log/{version}/",
    #                    threshold=i/100)
    #     # FNN
    #     # training(version= 'data_aug_0.8_2')
    #     evaluation(f"{config['default']['augmentation']}/{version}.pt", is_render=False, indexing=True,
    #                save_fig=True, save_samples=False)
    # version = '1_40k_07'
    version = '1_6000_3'


    # # mg = MeshAugmentation([], [])
    # # data = mg.load_samples(f'D:\meshingData\ANN\data_augmentation\\1\\training_samples_{version}.json')
    # # # data = mg.sampling(100, 0.7, f'D:\meshingData\ANN\data_augmentation\\1\\training_samples_{version}.json')
    # # mg.scatter_plot(data)
    #
    # # initial training
    #
    # data_path = f"{config['default']['augmentation']}\\1103\\training_samples_{version}.json"
    # data_path = f"{config['default']['augmentation']}\\1103\\training_samples_1_6000.json"
    # data_sampling(data_path, n=40000, threshold=0.7)

    # data_path = f"{config['default']['augmentation']}\\1\\training_samples_data_aug_2.8_1.json"
    # data_path = f"{config['default']['augmentation']}\\1\\data_aug.json"

    ## Meshing model training
    # start_time = time.time()
    # start_training(f"{config['default']['augmentation']}/{version}.pt",
    #                data_path,
    #                tensorboard_log=f"{config['default']['augmentation']}/log/{version}/")
    # print('Complete training in:', time.time() - start_time, 's.')

    model_path = f"{config['default']['augmentation']}/1_6000_2.pt"
    evaluation(model_path, version, is_render=False, indexing=False,
               save_fig=True, save_samples=False)

    # hyperparameter_search()

    # resampling
    # mg = MeshAugmentation([], [])
    # for i in range(0, 20, 5):
    #     version = f'data_aug_1.7_{0.7 + i/100}'
    #     data_sampling(f"{config['default']['augmentation']}\\1\\training_samples_{version}.json", n=10000,
    #                   threshold=0.7 + i/100)
    #     # mg.resampling(f"{config['default']['augmentation']}\\1\\training_samples_data_aug_0.7.json",
    #     #               target_name=f"{config['default']['augmentation']}\\1\\training_samples_{version}.json",
    #     #               n=i*1000)
    #     start_training(f"{config['default']['augmentation']}/{version}.pt",
    #                    f"{config['default']['augmentation']}\\1\\training_samples_{version}.json",
    #                    tensorboard_log=f"{config['default']['augmentation']}/log/{version}/")
    #
    #     evaluation(f"{config['default']['augmentation']}/{version}.pt", is_render=False, indexing=True,
    #                save_fig=True, save_samples=False)

    # For sample numbering testing
    # for i in [40, 100]: #5, 10
    #     version = f'data_aug_2.7_{i}'
    #     data_sampling(f"{config['default']['augmentation']}\\1\\training_samples_{version}.json", n=1000*i,
    #                   threshold=0.7)
    #     # mg.resampling(f"{config['default']['augmentation']}\\1\\training_samples_data_aug_0.7.json",
    #     #               target_name=f"{config['default']['augmentation']}\\1\\training_samples_{version}.json",
    #     #               n=i*1000)
    #     start_training(f"{config['default']['augmentation']}/{version}.pt",
    #                    f"{config['default']['augmentation']}\\1\\training_samples_{version}.json",
    #                    tensorboard_log=f"{config['default']['augmentation']}/log/{version}/")
    #
    #     evaluation(f"{config['default']['augmentation']}/{version}.pt", is_render=False, indexing=True,
    #                save_fig=True, save_samples=False)

    # FNN
    # training(version= 'data_aug_0.8_2')
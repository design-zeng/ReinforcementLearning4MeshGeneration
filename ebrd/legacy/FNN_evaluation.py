from general.mesh import MeshGeneration
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from sac.boundary_env import BoudaryEnv, boundary, read_polygon
import numpy as np
import general.data as data_process
import json
import matplotlib.pyplot as plt
import time
# from general.EBRD import Policy


# env = BoudaryEnv(boundary())
# env.boundary.show(style='k.-')
version = 'ea_experiments_0'
# version = 'ea_experiments_slim_8'

base_path = 'D:\\meshingData\\ANN\\'

device = torch.device("cuda")

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

mkdir_p(f'{base_path}plots/{version}/')
mkdir_p(f'{base_path}elements/{version}/')

def load_model(model_path):
    model = Policy().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_action(state, model):
    state = torch.FloatTensor(state).to(device)
    action, type_values = model(state)
    return action.tolist(), float(type_values)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 2 * (4 + 4)
        self.action_space = 2

        # self.affine1 = nn.Linear(self.state_space, 500)
        # self.action_head = nn.Linear(500, self.action_space)
        # self.type_head = nn.Linear(500, 1)
        num_nuerons = 64 # change from 64 to 128
        #
        # self.fc1 = nn.Linear(self.state_space, num_nuerons)
        # self.fc2 = nn.Linear(num_nuerons, num_nuerons)
        # self.action_head = nn.Linear(num_nuerons, self.action_space)
        # self.type_head = nn.Linear(num_nuerons, 1)

        self.fc1 = nn.Linear(self.state_space, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.action_head = nn.Linear(16, self.action_space)
        self.type_head = nn.Linear(16, 1)

        # self.fc1 = nn.Linear(self.state_space, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)
        # self.fc6 = nn.Linear(64, 32)
        # self.fc7 = nn.Linear(32, 16)
        # self.action_head = nn.Linear(16, self.action_space)
        # self.type_head = nn.Linear(16, 1)

        # self.fc1 = nn.Linear(self.state_space, 128)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, 16)
        # self.action_head = nn.Linear(16, self.action_space)
        # self.type_head = nn.Linear(16, 1)
        # self.fc1 = nn.Linear(self.state_space, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, 16)
        # self.action_head = nn.Linear(16, self.action_space)
        # self.type_head = nn.Linear(16, 1)

        # self.fc1 = nn.Linear(self.state_space, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, 16)
        # self.fc7 = nn.Linear(16, 8)
        # self.action_head = nn.Linear(8, self.action_space)
        # self.type_head = nn.Linear(8, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        # x = F.relu(self.affine1(x))
        # action = self.action_head(x)
        # state_values = self.type_head(x)
        # return action, state_values

        # x1 = self.fc1(x)
        # x5 = F.relu(self.fc2(x1))
        # action = self.action_head(x5)
        # state_values = self.type_head(x5)

        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        x5 = F.relu(self.fc5(x4))
        action = self.action_head(x5)
        state_values = self.type_head(x5)

        # x1 = self.fc1(x)
        # x2 = self.fc2(x1)
        # x3 = self.fc3(x2)
        # x4 = self.fc4(x3)
        # x5 = self.fc5(x4)
        # x6 = self.fc6(x5)
        # x7 = F.relu(self.fc7(x6))
        # action = self.action_head(x7)
        # state_values = self.type_head(x7)
        # x1 = self.fc1(x)
        # x2 = self.fc2(x1)
        # x3 = self.fc3(x2)
        # x4 = self.fc4(x3)
        # x5 = self.fc5(x4)
        # x7 = F.relu(self.fc6(x5))
        # action = self.action_head(x7)
        # state_values = self.type_head(x7)

        # x1 = self.fc1(x)
        # x2 = self.fc2(x1)
        # x3 = self.fc3(x2)
        # x4 = self.fc4(x3)
        # x5 = self.fc5(x4)
        # x6 = self.fc6(x5)
        # x7 = F.relu(self.fc7(x6))
        # action = self.action_head(x7)
        # state_values = self.type_head(x7)
        return action, state_values


def evaluation():
    # domains = [f'D:/python projects/meshgeneration/ui/domains/boundary{i}.json' for i in range(15, 15)]
    # # domains.append(f'D:/python projects/meshgeneration/ui/domains/test1.json')
    # domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary_hole_r2.json')
    domains = []
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary16.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary15.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/test1.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary4.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary8.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary9.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary10.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary_hole_r2.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary13.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/test2.json')
    domains.append(f'D:/python projects/meshgeneration/ui/domains/boundary13.json')
    models_static = [f'{base_path}models/ea_training_18/ebrd_model_{i}.pt' for i in range(1, 2)]
    # models_static = [f'{base_path}models/ea_bp_t2_2/ebrd_model_0.pt']
    # models_static = [f'{base_path}models/ebrd_model_256.pt']


    envs = [BoudaryEnv(read_polygon(name)) for name in domains]
    # envs = [BoudaryEnv(boundary())]
    # envs[0].boundary.show()
    # [print(e.boundary.poly_area()) for e in envs]

    models = [load_model(name) for name in models_static]

    for i, d in enumerate(envs):
        for j, m in enumerate(models):
            execution(f'domain{i}_model{j}', m, d)


def execution(name, model, env):
    max_steps = 10000
    running_reward = 10
    step = 0

    state, ep_reward = env.reset(), 0

    start = time.time()

    for i in range(max_steps):
        step += 1
        # state = state[
        action, type_values = get_action(state, model)
        # fig = plt.figure(figsize=(15,5))
        #
        # ax = fig.add_subplot(1, 3, 1)
        # [seg.show() for seg in env.boundary.all_segments()]
        state, reward, done, _ = env.move(action, round(type_values, 1), 0.9, 0.5)
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
        # print(i, reward, len(env.updated_boundary.vertices))
        ep_reward += reward
        if done:
            break


    execution_time = time.time() - start
    print(f"Execution time: {execution_time}s.")
    print(len(env.generated_meshes), len(env.generated_meshes) / execution_time)
    # env.boundary.show('b-')

    # env.smooth(env.boundary.vertices)
    # env.smooth_pave(env.boundary.vertices, env.updated_boundary.vertices, iteration=400)
    # env.plot_meshes(env.generated_meshes, quality=True, type=1)

    # env.write_generated_elements_2_file(f"{base_path}elements/{version}/elements_{name}")

    # samples, output_types, outputs = env.extract_samples(env.generated_meshes)
    # print(len(env.generated_meshes), len([m for m in env.generated_meshes if env.get_quality(m, index=1) > 0.7]), len(samples))

    # env.save_meshes(env.generated_meshes[:9], quality=True)
    # env.plot_meshes(env.generated_meshes[:9], quality=True)
    # env.boundary.show()

    # env.save_meshes(f"{base_path}plots/{version}/{i_episode}.png", env.generated_meshes, quality=True,
    #                 indexing=True,
    #                 type=1, dpi=300)
    # env.boundary.savefig(f"{base_path}plots/{version}/{name}.png", style='k-', dpi=1000)
    print("Figure saved!")

evaluation()
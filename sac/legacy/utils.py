from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, D1, h1, h2, outputs):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(D1, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, outputs)
        )

    def forward(self, x):
        return self.net(x)


def draw(scores, path="fig.png", title="Performance", xlabel="Episode #", ylabel="Score"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path)


def plot(file_name):
    # res = []
    with open(file_name, 'r+') as fr:
        res = json.load(fr)

    # print()
    # plt.plot(range(500), res[:500])
    # plt.ylim(-10, 100)
    # plt.show()

    r = pd.DataFrame([[i, d] for i, d in enumerate(res)],
        columns=['Epoch', 'Reward'])

    ss = list(zip(range(len(res)), res))
    tt = sorted(ss, key=lambda k: k[1], reverse=True)

    bad_samples = []
    max_avg = -999999

    for i in range(100, len(res)):
        avg = sum(res[i - 100: i]) / 100

        if max_avg < avg:
            max_avg = avg

        last_avg = sum(res[i - 99: i + 1]) / 100
        if last_avg < max_avg:
            bad_samples.append((i, res[i], math.fabs((max_avg - last_avg) / max_avg)))

    bad_samples = []
    max_avg = -999999

    x = []
    m_v = []
    l_v = []

    _m_v = []
    _l_v = []

    max = []
    current = []

    l = []

    for i in range(100, len(res)):
        avg = sum(res[i - 100: i]) / 100

        if max_avg < avg:
            max_avg = avg

        last_avg = sum(res[i - 99: i + 1]) / 100
        if last_avg < max_avg and res[i] > 10:
            bad_samples.append((i, res[i], math.fabs((max_avg - last_avg) / max_avg)))
            x.append(i)
            m_v.append(math.fabs((max_avg - last_avg) / max_avg))
            l_v.append(math.fabs((max_avg - last_avg) / last_avg))
            _m_v.append(math.fabs((avg - last_avg) / avg))
            _l_v.append(math.fabs((avg - last_avg) / last_avg))

            max.append(max_avg)
            current.append(res[i])
            l.append(last_avg)

    plt.plot(x, m_v, 'r--', x, _m_v, 'g-', x, _l_v, 'k--')  # x, l_v, 'b--',
    # sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
    plt.ylim(-1, 2)
    plt.show()

# plot("A2C/plots/test_58/total_rewards.txt")
# plot("D:\\meshingData\\A2C\\plots\\test_58\\total_rewards.txt")

def write_csv(data, file_name):
    r = pd.DataFrame(data, columns=['state', 'action', 'reward', 'done', 'state value'])
    r.to_csv(file_name)

def plot_rewarding(data, save=False, name=None):
    quality = []
    boundary = []
    area = []

    rewards = [0]
    avg = []

    [(quality.append(d[0]), boundary.append(d[1]), area.append(d[2]),
      avg.append(d[4])) for i, d in enumerate(data)]
    # *(1 - math.log(0.04 * i + 1))
    x = range(len(quality))
    mean = [sum(avg[:i+1])/(i + 1) for i in range(len(avg))]
    diff = [math.fabs(avg[i] - mean[i]) for i in range(len(avg))]

    c_diff = [sum(diff[:i+1]) for i in range(len(avg))]

    # [rewards.append(rewards[-1] + math.sqrt(d[0] * d[1]) - (1 - d[2]) * 1.5) for i, d in enumerate(data)]
    [print((d[2]/60 - d[4])/(d[2]/60)) for i, d in enumerate(data)]

    [rewards.append(rewards[-1] + math.sqrt(d[0] * d[1]) - ((d[2]/60 - d[4])/(d[2]/60))
                                                            if d[2]/60 - d[4] > 0 else 0) for i, d in enumerate(data)]
    # penalty = [(area[i] - ((area[i-1] - 1)/i)*(i+1) - 1)*100 for i in range(1, len(area))]
    # penalty.insert(0, 0)

    if save and name:
        plt.clf()
        plt.plot(x, quality, 'r--', x, boundary, 'b--', x, area, 'g-', x, rewards[1:], 'ko')
        # plt.gca().set_aspect('equal', adjustable='box') x, penalty, 'm--'
        # plt.plot(x, quality, 'r--', x, boundary, 'b--', x, area, 'g-', x, mean, 'k--', x, diff, 'y--')
        plt.savefig(name, dpi=300)
        plt.close('all')
    else:
        plt.plot(x, quality, 'r--', x, boundary, 'b--', x, area, 'g-', x, rewards[1:], 'ko')
        plt.show()

def read_json(file_name):
    with open(file_name, 'r') as fr:
        data = json.loads(fr.read())

    print(data)

# read_json("C:\\Users\\Jay\\Downloads\\ebrd_7.json")
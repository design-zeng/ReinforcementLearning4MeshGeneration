import math

import matplotlib.pyplot as plt


def plot_sample(state, action):
    L = len(state)
    N = int((L - 6) / 2)
    left = [-i - 2 for i in reversed(range(0, N, 2))]
    right = [i for i in range(0, N, 2)]
    all = left + right

    medium = [N + i for i in range(0, 6, 2)]

    n_x = [state[j] * math.cos(state[j + 1]) for j in all]
    n_x.insert(int(N / 2), 0)
    r_x = [state[j] * math.cos(state[j + 1]) for j in medium]
    n_y = [state[j] * math.sin(state[j + 1]) for j in all]
    n_y.insert(int(N / 2), 0)
    r_y = [state[j] * math.sin(state[j + 1]) for j in medium]
    plt.plot(n_x, n_y, 'k.-')
    plt.plot(r_x, r_y, 'y.-')
    target_x, target_y = action[1] * math.cos(action[2]), action[1] * math.sin(action[2])
    plt.plot(target_x, target_y, 'r.')
    plt.title(f'type: {action[0]}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

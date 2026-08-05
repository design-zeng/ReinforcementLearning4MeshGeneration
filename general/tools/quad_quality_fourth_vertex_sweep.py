import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from general.components import Mesh, Vertex
from general.component_plotting import plot_point


sns.set_theme(style="whitegrid")


def compute():
    v1 = Vertex(0, 0)
    v2 = Vertex(-1, 0.5)
    v3 = Vertex(3.4, 0.2)
    xs = np.arange(-10, 10, 0.1)
    ys = np.arange(1, 11, 0.1)
    vs = [Vertex(x, y) for x in xs for y in ys]

    for v in [v1, v2, v3]:
        plot_point(v)

    angle = []
    ratio = []
    quality = []
    for v in vs:
        mesh = Mesh([v1, v2, v, v3])
        if mesh.is_valid(quality_method=0):
            q1, q2 = mesh.get_quality_3()
            quality.append(math.pow(q1*q2, 1/2))
            angle.append(v.to_find_clockwise_angle(v3, v2))
            ratio.append(v.distance_to(v2)/v.distance_to(v3))

    # x, y = np.meshgrid(np.array(angle), np.array(ratio))
    # z = np.tile(quality, (len(angle), 1))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(angle, ratio, quality, marker='.')

    plt.show()

if __name__ == "__main__":
    compute()

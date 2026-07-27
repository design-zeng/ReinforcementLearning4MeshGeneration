from general.components import Mesh, Vertex, solve_quadratic_equation
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="whitegrid")


def compute():
    v1 = Vertex(0, 0)
    v2 = Vertex(-1, 0.5)
    v3 = Vertex(3.4, 0.2)
    xs = np.arange(-10, 10, 0.1)
    ys = np.arange(1, 11, 0.1)
    vs = [Vertex(x, y) for x in xs for y in ys]

    for v in [v1, v2, v3]:
        v.show()

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

    x, y = np.meshgrid(np.array(angle), np.array(ratio))
    z = np.tile(quality, (len(angle), 1))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(angle, ratio, quality, marker='.')

    plt.show()


def find_perpendicular_verctor(v1, v2):
    u = (v1.x - v2.x, v1.y - v2.y)

    pass

# compute()
# v2 = Vertex(-11, 5)
# v3 = Vertex(3.4, 0.2)
# vv = v2.get_perpendicular_vertex(v3)
#
# for v in [v2, v3]:
#     v.show()
#
# for v in [vv[0], vv[1]]:
#     v.show('r.')
#
# print(vv[0].to_find_clockwise_angle(v2, v3), vv[1].to_find_clockwise_angle(v2, v3))
# plt.show()
# print()

def test_plot():

    data = pd.DataFrame(data={'x': [1, 2, 3], 'y': [1, 2,3]})
    # Plot the responses for different events and regions
    sns.lineplot(x="x", y="y",
                 data=data)
    plt.show()
# test_plot()
from matplotlib import pyplot
#from shapely.geometry import Polygon
#from descartes.patch import PolygonPatch

from sac.figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
from general.components import *
from general.mesh import connect_vertices
import json


def plot_polygon():
    fig = pyplot.figure(1, figsize=SIZE, dpi=90)

    # 1: valid polygon
    ax = fig.add_subplot(121)

    ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
    # int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
    polygon = Polygon(ext)

    # plot_coords(ax, polygon.interiors[0])
    plot_coords(ax, polygon.exterior)

    patch = PolygonPatch(polygon, facecolor=color_isvalid(polygon), edgecolor=color_isvalid(polygon, valid=BLUE), alpha=0.5, zorder=2)
    ax.add_patch(patch)

    ax.set_title('a) valid')

    set_limits(ax, -1, 3, -1, 3)

    #2: invalid self-touching ring
    # ax = fig.add_subplot(122)
    # ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
    # int = [(1, 0), (0, 1), (0.5, 1.5), (1.5, 0.5), (1, 0)][::-1]
    # polygon = Polygon(ext, [int])
    #
    # plot_coords(ax, polygon.interiors[0])
    # plot_coords(ax, polygon.exterior)
    #
    # patch = PolygonPatch(polygon, facecolor=color_isvalid(polygon), edgecolor=color_isvalid(polygon, valid=BLUE), alpha=0.5, zorder=2)
    # ax.add_patch(patch)
    #
    # ax.set_title('b) invalid')

    set_limits(ax, -1, 3, -1, 3)

    pyplot.show()

def gen_boundary():
    # p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13
    points = [(0, 0), (0, 6), (12, 6), (12, 0), (17, 0), (17, -5), (11, -5),
                                                         (10, -11), (-3, -12), (-3, -7), (-6, -9), (-12, -1)]
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = [Vertex(p[0], p[1]) for p in points]
    # for v in vertices:
    #     v.show()
    # plt.show()
    vertices = []
    vertices.extend(p1.sampling_between_endpoints(p2, 5, is_even=True))
    vertices.extend(p2.sampling_between_endpoints(p3, 5, is_even=True))
    vertices.extend(p3.sampling_between_endpoints(p4, 5, is_even=True))
    vertices.extend(p4.sampling_between_endpoints(p5, 5, is_even=True))
    vertices.extend(p5.sampling_between_endpoints(p6, 5, is_even=True))
    vertices.extend(p6.sampling_between_endpoints(p7, 5, is_even=True))
    vertices.extend(p7.sampling_between_endpoints(p8, 5, is_even=True))
    vertices.extend(p8.sampling_between_endpoints(p9, 5, is_even=True))
    vertices.extend(p9.sampling_between_endpoints(p10, 5, is_even=True))
    vertices.extend(p10.sampling_between_endpoints(p11, 3, is_even=True))
    vertices.extend(p11.sampling_between_endpoints(p12, 5, is_even=True))
    vertices.extend(p12.sampling_between_endpoints(p1, 5, is_even=True))
    [v.show() for v in vertices]
    # plt.show()
    connect_vertices(vertices)
    env = Boundary2D(vertices)
    return env

def boundary(index=0):

    if index == 0:
        points = [Vertex(0, 1), Vertex(0, 2), Vertex(0, 3), Vertex(0, 4), Vertex(0, 5), Vertex(0, 6),
                  Vertex(1, 6), Vertex(2, 6), Vertex(3, 6), Vertex(4, 6), Vertex(5, 6), Vertex(6, 6),
                  Vertex(7, 5), Vertex(8, 4), Vertex(9, 3), Vertex(10, 2), Vertex(11, 1), Vertex(12, 0),
                  Vertex(11, -1), Vertex(10, -2), Vertex(9, -3), Vertex(8, -4), Vertex(7, -5), Vertex(6, -6),
                   Vertex(5, -5), Vertex(4, -4), Vertex(3, -3), Vertex(2, -2), Vertex(1, -1), Vertex(0, 0)]
    elif index == 1:
        points = [Vertex(0, 1), Vertex(0, 2), Vertex(0, 3), Vertex(0, 4), Vertex(0, 5), Vertex(0, 6), Vertex(0, 7),
                  Vertex(0, 8), Vertex(0, 9), Vertex(0, 10), Vertex(0, 11),
                  Vertex(1, 11), Vertex(2, 11), Vertex(3, 11), Vertex(4, 11), Vertex(5, 11), Vertex(6, 11), Vertex(7, 11),
                  Vertex(8, 11),
                  Vertex(9, 11), Vertex(8, 10), Vertex(7.5, 9), Vertex(7, 8), Vertex(6.5, 7), Vertex(6, 6),
                  Vertex(6.5, 5.5),
                  Vertex(7, 5), Vertex(8, 4), Vertex(9, 3), Vertex(10, 2), Vertex(11, 1), Vertex(12, 0),
                  Vertex(11, -1), Vertex(10, -2), Vertex(9, -3), Vertex(8, -4), Vertex(7, -5), Vertex(6, -6),
                  Vertex(5, -5), Vertex(4, -4), Vertex(3, -3), Vertex(2, -2), Vertex(1, -1), Vertex(0, 0)]
    elif index == 2:
        points = [Vertex(0, 1), Vertex(0, 2), Vertex(0, 3), Vertex(0, 4), Vertex(1, 4), Vertex(2, 4), Vertex(3, 4), Vertex(4, 4),
                  Vertex(5, 4), Vertex(5, 3), Vertex(5, 2), Vertex(5, 1), Vertex(5, 0), Vertex(5, -1), Vertex(5, -2), Vertex(5, -3),
                  Vertex(5, -4), Vertex(5, -5), Vertex(5, -6), Vertex(5, -7), Vertex(4, -7), Vertex(3, -7), Vertex(2, -7), Vertex(1, -7),
                  Vertex(0, -7), Vertex(0, -6), Vertex(0, -5), Vertex(0, -4), Vertex(1, -4), Vertex(2, -4), Vertex(3, -4), Vertex(4, -4),
                  Vertex(4, -3), Vertex(4, -2), Vertex(4, -1), Vertex(4, 0), Vertex(3, 0),
                  Vertex(2, 0), Vertex(1, 0), Vertex(0, 0)]
    elif index == -1:
        points = [Vertex(0, 1), Vertex(0, 2), Vertex(0, 3), Vertex(0, 4), Vertex(0, 5), Vertex(0, 6),
                  Vertex(1, 6), Vertex(2, 6), Vertex(3, 6), Vertex(4, 6), Vertex(5, 6), Vertex(6, 6),
                  Vertex(6, 5), Vertex(6, 4), Vertex(6, 3), Vertex(6, 2), Vertex(6, 1), Vertex(6, 0),
                   Vertex(5, 0), Vertex(4, 0), Vertex(3, 0), Vertex(2, 0), Vertex(1, 0), Vertex(0, 0)]
    connect_vertices(points)
    env = Boundary2D(points)
    return env

def read_polygon(filename):
    with open(filename, 'r') as fr:
        vertices = json.loads(fr.readline())
        # vertices = json.loads(fr.readline())
    points = [Vertex(p[0]/100, p[1]/100) for p in vertices]
    connect_vertices(points)
    env = Boundary2D(points)
    return env

def get_boundary_data(boundary):
    xs = [v.x for v in boundary.vertices]
    xs.append(xs[0])
    ys = [v.y for v in boundary.vertices]
    ys.append(ys[0])
    return xs, ys

def experiement_boundaries():
    d1 = read_polygon('../ui/domains/boundary16.json')
    d2 = read_polygon('../ui/domains/boundary15.json')
    d3 = read_polygon('../ui/domains/test1.json')
    d4 = read_polygon('../ui/domains/test2.json')

    plt.figure(1)
    ax1 = plt.subplot(221)
    d1_xs, d1_ys = get_boundary_data(d1)
    ax1.set_title("D1")
    plt.axis('off')
    # ax1.set_aspect(4)
    plt.plot(d1_xs, d1_ys, 'k.-')
    print(len(d1_xs))

    ax2 = plt.subplot(223)
    d2_xs, d2_ys = get_boundary_data(d2)
    ax2.set_title("D2")
    plt.plot(d2_xs, d2_ys, 'k.-')
    plt.axis('off')
    # ax2.set_aspect(4)
    print(len(d2_xs))

    ax3 = plt.subplot(122)
    d3_xs, d3_ys = get_boundary_data(d3)
    ax3.set_title("D3")
    plt.plot(d3_xs, d3_ys, 'k.-')
    # ax3.set_aspect(4)
    plt.axis('off')
    print(len(d3_xs))

    # ax4 = plt.subplot(144)
    # d4_xs, d4_ys = get_boundary_data(d4)
    # ax4.set_title("D4")
    # # ax4.set_aspect(4)
    # plt.plot(d4_xs, d4_ys, 'b.-')
    # plt.axis('off')
    # print(len(d4_xs))

    plt.show()
    plt.savefig("boundary_domains.png")


# env = gen_boundary()
# env.show()
# env = read_polygon('../ui/domains/problem.json')
# env = boundary(2)
# env.show('k.-')
# experiement_boundaries()
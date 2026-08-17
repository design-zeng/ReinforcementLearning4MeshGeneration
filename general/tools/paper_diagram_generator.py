import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from general.geometry import Vertex, Segment, Quad
from general.quality import QuadQuality
from general.mesh_io import read_polygon
from general.plotting import plot_segment


output_path = Path(__file__).parent.parent.parent / "general" / "output"
domains_path = Path(__file__).parent.parent.parent / "samples" / "domains"
# mesh snapshots are produced by sac/infer.py evaluation() (save_fig=True)
img_path = Path(__file__).parent.parent.parent / "sac" / "output" / "evaluation"


def bad_cases():
    ax1 = plt.subplot(131)
    ax1.plot([1.5, 1, 2, 1.8, 1.5], [1, 2, 1, 4, 1], 'b-o')
    ax1.set_title("(a)")

    ax2 = plt.subplot(132)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [Vertex(1, 1), Vertex(2, 0.9), Vertex(2.8, 0.8), Vertex(3, 1.9), Vertex(2.2, 2), Vertex(3.5, 1.2), Vertex(2.2, 2.2), Vertex(3.6, 2.3), Vertex(4.5, 1.3), Vertex(5.5, 1.4)]

    segmts = []
    segmts.append(Segment(p1, p2))
    segmts.append(Segment(p2, p3))
    segmts.append(Segment(p3, p4))
    segmts.append(Segment(p4, p5))
    segmts.append(Segment(p5, p2))
    segmts.append(Segment(p3, p6))
    segmts.append(Segment(p6, p7))
    segmts.append(Segment(p7, p8))
    segmts.append(Segment(p6, p9))
    segmts.append(Segment(p8, p9))
    segmts.append(Segment(p9, p10))
    for s in segmts:
        plot_segment(s)
    ax2.set_title("(b)")

    ax3 = plt.subplot(133)
    p1, p2, p3, p4, p5, p6, p7, p8 = [Vertex(1, 1), Vertex(2, 0.9), Vertex(2.8, 0.8), Vertex(2.8, 1.6), Vertex(2.2, 1.7), Vertex(3.5, 1.2), Vertex(2.2, 2.2), Vertex(3.2, 2.3),]

    segmts = []
    segmts.append(Segment(p1, p2))
    segmts.append(Segment(p2, p3))
    segmts.append(Segment(p3, p4))
    segmts.append(Segment(p4, p5))
    segmts.append(Segment(p5, p2))
    segmts.append(Segment(p3, p6))
    segmts.append(Segment(p6, p7))
    segmts.append(Segment(p7, p8))
    for s in segmts:
        plot_segment(s)
    ax3.set_title("(c)")

    plt.show()


def primitive_rules():
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot([1, 1.5, 2.5, 3], [1, 1.5, 1.5, 1], 'b-o')
    ax1.plot([1.5, 1.5, 2.5, 2.5], [1.5, 2.5, 2.5, 1.5], 'b--')
    ax1.set_title("(a)")

    ax2 = fig.add_subplot(132)
    ax2.plot([0.9, 1.4, 1.5, 2.5, 3], [3, 2.6, 1.5, 1.5, 1], 'b-o')
    ax2.plot([1.4, 2.7, 2.5], [2.6, 2.4, 1.5], 'b--')
    ax2.set_title("(b)")

    ax3 = fig.add_subplot(133)
    ax3.plot([0.9, 1.4, 1.5, 2.5, 2.7, 3.2], [3, 2.6, 1.5, 1.5, 2.4, 2.8], 'b-o')
    ax3.plot([1.4, 2.7], [2.6, 2.4], 'b--')
    ax3.set_title("(c)")

    for a in fig.axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    plt.show()


def generation_partial_boundary():
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(-1, 6.5)
    ax2.set_ylim(-.5, 6)
    ax2.plot([0.3, 1.4, 1.5, 2.5, 3.5, 5.1, 3.4, 3.5, 2.2, 1, 0.5, -0.6, 0.3], [3.4, 2.6, 1.5, 1.5, 0.5, 1.4, 2.9, 4.3, 5.45, 5.7, 4.6, 4, 3.4], 'k-o')
    circle2 = patches.Circle((1.5, 1.5), 4, color='black', linestyle='--', fill=False)
    ax2.add_artist(circle2)
    ax2.plot([1.5, 1.15], [1.5, 5.57], color='black', linestyle='--', lw=1)
    ax2.plot([1.5, 3.28], [1.5, 5.17], color='black', linestyle='--', lw=1)
    ax2.plot([1.5, 4.93], [1.5, 3.67], color='black', linestyle='--', lw=1)
    ax2.plot([1.5, 5.6], [1.5, 1.54], color='black', linestyle='--', lw=1)
    # ax2.plot([1.4, 2.7, 2.5], [2.6, 2.4, 1.5], 'b--')
    ax2.set_frame_on(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def coordinate_system():
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(-1.5, 4.5)
    ax2.set_ylim(-1.5, 4.5)
    ax2.plot([0.3, 1.4, 1.5, 2.5, 3.5], [3.4, 2.6, 1.5, 1.9, 0.5], 'b-o')
    # ax2.annotate("", xy=(0, 4), xytext=(0, 0), arrowprops = dict(arrowstyle="->"))
    # ax2.annotate("", xy=(4, 0), xytext=(0, 0), arrowprops = dict(arrowstyle="->"))
    plt.arrow(0, 0, 0, 4, width=0.0015, color="k", clip_on=False, head_width=0.12, head_length=0.12)
    plt.arrow(0, 0, 4, 0, width=0.0015, color="k", clip_on=False, head_width=0.12, head_length=0.12)

    plt.arrow(1.5, 1.5, -0.7, 2, width=0.015, color="k", clip_on=False, head_width=0.12, head_length=0.12)
    plt.arrow(1.5, 1.5, 1.724, 0.7, width=0.015, color="k", clip_on=False, head_width=0.12, head_length=0.12)

    # ax2.set_yticklabels([])
    # ax2.set_xticklabels([])
    # ax2.plot([1.4, 2.7, 2.5], [2.6, 2.4, 1.5], 'b--')
    # ax = fig.add_subplot(122)
    # ax.set_xlim(-1.5, 4.5)
    # ax.set_ylim(-1.5, 4.5)
    # ax.plot([-1.2, -0.1, 0, 1, 2], [1.9, 1.1, 0, 0, -1], 'b-o')

    plt.show()


def output_types():
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot([0.9, 1.4, 1.5, 2.5, 3], [3, 2.6, 1.5, 1.5, 1], 'b-')
    ax1.plot([1.4, 2.7, 2.5], [2.6, 2.4, 1.5], 'b--')
    ax1.plot([2.7], [2.4], 'o')
    ax1.set_title("(a) type 0")

    ax2 = fig.add_subplot(132)
    ax2.plot([0.9, 1.4, 1.5, 2.5, 2.7, 3.2], [3, 2.6, 1.5, 1.5, 2.4, 2.8], 'b-')
    ax2.plot([1.4, 2.7], [2.6, 2.4], 'b--')
    ax2.plot([1.4], [2.6], 'o')
    ax2.set_title("(b) type 1")

    ax3 = fig.add_subplot(133)
    ax3.plot([0.9, 1.4, 1.5, 2.5, 2.7, 3.2], [3, 2.4, 1.5, 1.5, 2.6, 2.8], 'b-')
    ax3.plot([1.4, 2.7], [2.4, 2.6], 'b--')
    ax3.plot([2.7], [2.6], 'o')
    ax3.set_title("(c) type 2")

    for a in fig.axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    plt.show()


def element_quality_sweep():
    v1 = Vertex(0, 0)
    v2 = Vertex(-1, 0.5)
    v3 = Vertex(3.4, 0.2)
    xs = np.arange(-10, 10, 0.1)
    ys = np.arange(1, 11, 0.1)
    vs = [Vertex(x, y) for x in xs for y in ys]

    for v in [v1, v2, v3]:
        plt.plot(v.x, v.y, 'b.')

    angle = []
    ratio = []
    quality = []
    for v in vs:
        quad = Quad([v1, v2, v, v3])
        if quad.is_valid():
            quality.append(QuadQuality.edge_angle(quad))
            angle.append(v.clockwise_angle(v3, v2))
            ratio.append(v.distance_to(v2) / v.distance_to(v3))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(angle, ratio, quality, marker='.')
    plt.show()


def experiment_boundaries():
    def boundary_data(boundary):
        xs = [v.x for v in boundary.vertices]
        xs.append(xs[0])
        ys = [v.y for v in boundary.vertices]
        ys.append(ys[0])
        return xs, ys

    d1 = read_polygon(domains_path / "boundary16.json")
    d2 = read_polygon(domains_path / "boundary15.json")
    d3 = read_polygon(domains_path / "test1.json")

    plt.figure(1)
    ax1 = plt.subplot(221)
    d1_xs, d1_ys = boundary_data(d1)
    ax1.set_title("D1")
    plt.axis('off')
    plt.plot(d1_xs, d1_ys, 'k.-')

    ax2 = plt.subplot(223)
    d2_xs, d2_ys = boundary_data(d2)
    ax2.set_title("D2")
    plt.plot(d2_xs, d2_ys, 'k.-')
    plt.axis('off')

    ax3 = plt.subplot(122)
    d3_xs, d3_ys = boundary_data(d3)
    ax3.set_title("D3")
    plt.plot(d3_xs, d3_ys, 'k.-')
    plt.axis('off')

    plt.show()


def read_img():
    # Assemble a panel from the mesh figures produced by sac.infer.
    imgs = sorted(img_path.rglob("*.png"))[:6]
    if not imgs:
        raise FileNotFoundError(
            f"No mesh figures under {img_path}. Run `python -m sac.infer` (save_fig=True) first.")
    fig = plt.figure()
    for k, im in enumerate(imgs):
        ax = fig.add_subplot(2, 3, k + 1)
        ax.imshow(plt.imread(im))
        ax.set_title(f"({chr(97 + k)})")
    for a in fig.axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/e.png", dpi=1000)

if __name__ == "__main__":
    bad_cases()
    # env.boundary.show()
    # primitive_rules()
    # generation_partial_boundary()
    # coordinate_system()
    # output_types()
    # read_img()
    # element_quality_sweep()
    # experiment_boundaries()

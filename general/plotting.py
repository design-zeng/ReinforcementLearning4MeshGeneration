import math

import matplotlib.pyplot as plt
import seaborn as sns

from general.geometry import Segment
from general.mesh import edge_angle_quality


def plot_segment(segment, style='b.-', linewidth=2, markersize=0.1):
    plt.plot([segment.point1.x, segment.point2.x], [segment.point1.y, segment.point2.y], style,
             linewidth=linewidth, markersize=markersize)
    plt.gca().set_aspect('equal', adjustable='box')


def show_boundary(boundary, style='b.-', linewidth=1, markersize=6, show=True):
    segts = boundary.all_segments()
    for segt in segts:
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue
        plot_segment(segt, style=style, linewidth=linewidth, markersize=markersize)
    plt.gca().set_aspect('equal', adjustable='box')
    if show:
        plt.show()


def plot_boundary(boundary, style='b.-', linewidth=2, markersize=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    segts = boundary.all_segments()
    x, y = [], []
    for segt in segts:
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue
        plot_segment(segt, style=style, linewidth=linewidth, markersize=markersize)
        x.extend([segt.point1.x, segt.point2.x])
        y.extend([segt.point1.y, segt.point2.y])

    ax.set_frame_on(False)
    plt.gca().set_xlim((min(x) - 0.1, max(x) + 0.1))
    plt.gca().set_ylim((min(y) - 0.1, max(y) + 0.1))
    plt.xticks([])
    plt.yticks([])


def savefig_boundary(boundary, name, title="", style="k.-", dpi=600):
    sns.set_context('paper')
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title(title)
    segts = boundary.all_segments()
    x, y = [], []
    for segt in segts:
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue
        plot_segment(segt, style=style, linewidth=2, markersize=10)
        x.extend([segt.point1.x, segt.point2.x])
        y.extend([segt.point1.y, segt.point2.y])

    ax.set_frame_on(False)
    plt.gca().set_xlim((min(x) - 0.1, max(x) + 0.1))
    plt.gca().set_ylim((min(y) - 0.1, max(y) + 0.1))
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)

    plt.savefig(name, dpi=dpi)
    plt.close('all')


def show_quad(quad, quality=3):
    for i in range(len(quad.vertices)):
        segt = Segment(quad.vertices[i], quad.vertices[i - 1])
        plot_segment(segt, style='k.-', linewidth=1, markersize=8)
    plt.gca().set_aspect('equal', adjustable='box')
    if quality != 0:
        center = quad.get_centroid()
        q1, q2 = edge_angle_quality(quad)
        _quality = round(math.sqrt(q1 * q2), 2)
        plt.text(center.x * 0.8, center.y * 0.8, str(_quality), fontsize=15)
    plt.gca().set_frame_on(False)
    plt.xticks([])
    plt.yticks([])


def render_boundary(boundary):
    plt.clf()
    show_boundary(boundary, style='b-', show=False)
    plt.pause(0.001)


def close_render():
    plt.close('all')


def generate_meshes_canvas(mesh_gen, quads, quality, indexing, quality_index, style):
    plot_boundary(mesh_gen.boundary, style=style, linewidth=1)
    for idx, m in enumerate(quads):
        center = m.get_centroid(diff=True)
        if quality and indexing:
            _quality = round(mesh_gen.get_quality(mesh_gen.boundary, element=m, index=quality_index), 4)
            plt.text(center.x, center.y, f"{idx}; {_quality}", fontsize=6)
        elif quality:
            _quality = round(mesh_gen.get_quality(mesh_gen.boundary, element=m, index=quality_index), 4)
            plt.text(center.x, center.y, str(_quality), fontsize=6)
        elif indexing:
            plt.text(center.x, center.y, str(idx), fontsize=4)


def save_meshes(mesh_gen, name, quads, quality=False, indexing=False, quality_index=0, dpi=300, style='k.-'):
    plt.clf()
    generate_meshes_canvas(mesh_gen, quads, quality, indexing, quality_index, style=style)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
    plt.savefig(name, dpi=dpi)
    plt.close('all')

import matplotlib.pyplot as plt
import seaborn as sns

from general.geometry import Segment
from general.quality import edge_angle_quality, robust_quality


def plot_segment(segment, style='b.-', linewidth=2, markersize=0.1):
    plt.plot([segment.point1.x, segment.point2.x], [segment.point1.y, segment.point2.y], style,
             linewidth=linewidth, markersize=markersize)
    plt.gca().set_aspect('equal', adjustable='box')


def _plot_own_segments(boundary, style, linewidth, markersize):
    x, y = [], []
    for segt in boundary.own_segments():
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue
        plot_segment(segt, style=style, linewidth=linewidth, markersize=markersize)
        x.extend([segt.point1.x, segt.point2.x])
        y.extend([segt.point1.y, segt.point2.y])
    return x, y


def _frame_canvas(x, y):
    plt.gca().set_frame_on(False)
    plt.gca().set_xlim((min(x) - 0.1, max(x) + 0.1))
    plt.gca().set_ylim((min(y) - 0.1, max(y) + 0.1))
    plt.xticks([])
    plt.yticks([])


def plot_boundary(boundary, style='b.-', linewidth=2, markersize=10):
    plt.figure().add_subplot(111)
    x, y = _plot_own_segments(boundary, style, linewidth, markersize)
    _frame_canvas(x, y)


def savefig_boundary(boundary, name, title="", style="k.-", dpi=600):
    sns.set_context('paper')
    plt.figure().add_subplot(111).set_title(title)
    x, y = _plot_own_segments(boundary, style, 2, 10)
    _frame_canvas(x, y)
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
        plt.text(center.x * 0.8, center.y * 0.8, str(round(edge_angle_quality(quad), 2)), fontsize=15)
    plt.gca().set_frame_on(False)
    plt.xticks([])
    plt.yticks([])


def render_boundary(boundary):
    plt.clf()
    _plot_own_segments(boundary, 'b-', 1, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.001)


def close_render():
    plt.close('all')


def generate_meshes_canvas(mesh_gen, quads, quality, indexing, metric, style):
    plot_boundary(mesh_gen.boundary, style=style, linewidth=1)
    for idx, quad in enumerate(quads):
        center = quad.get_centroid(diff=True)
        if quality:
            label = f"{idx}; " if indexing else ""
            plt.text(center.x, center.y, label + str(round(metric(quad), 4)), fontsize=6)
        elif indexing:
            plt.text(center.x, center.y, str(idx), fontsize=4)


def save_meshes(mesh_gen, name, quads, quality=False, indexing=False, metric=robust_quality, dpi=300, style='k.-'):
    plt.clf()
    generate_meshes_canvas(mesh_gen, quads, quality, indexing, metric, style=style)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
    plt.savefig(name, dpi=dpi)
    plt.close('all')

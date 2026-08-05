import math

import matplotlib.pyplot as plt
import seaborn as sns

from general.components import Segment


def plot_point(point, style='b.'):
    plt.plot(point.x, point.y, style)
    plt.gca().set_aspect('equal', adjustable='box')


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


def save_intermediate_boundary_fig(boundary, name, boundary_vs, title="", style="k.-", dpi=300, r_vertices=None):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    segts = boundary.all_segments()
    x, y = [], []
    for segt in segts:
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue
        x.extend([segt.point1.x, segt.point2.x])
        y.extend([segt.point1.y, segt.point2.y])

        if r_vertices is not None:
            if segt.point1 in r_vertices and segt.point2 in r_vertices:
                plot_segment(segt, style="b.-", linewidth=2, markersize=10)
            else:
                plot_segment(segt, style=style, linewidth=2, markersize=10)

            if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
                plot_segment(segt, style="r.-", linewidth=2, markersize=10)
        else:
            if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
                plot_segment(segt, style="r.-", linewidth=2, markersize=10)
            else:
                plot_segment(segt, style=style, linewidth=2, markersize=10)

    ax.set_frame_on(False)
    plt.gca().set_xlim((min(x) - 0.1, max(x) + 0.1))
    plt.gca().set_ylim((min(y) - 0.1, max(y) + 0.1))
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(name, dpi=dpi)
    plt.close('all')


def save_vertices_into_fig(boundary, name, boundary_vs, title="", style="k.-", dpi=300):
    sns.set_context('paper')
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    segts = boundary.all_segments()
    x, y = [], []
    for segt in segts:
        if segt.point1 not in boundary.vertices or segt.point2 not in boundary.vertices:
            continue

        if segt.point1 in boundary_vs and segt.point2 in boundary_vs:
            plot_segment(segt, style=style, linewidth=1, markersize=6)
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


def show_mesh(mesh, quality=3):
    for i in range(len(mesh.vertices)):
        segt = Segment(mesh.vertices[i], mesh.vertices[i - 1])
        plot_segment(segt, style='k.-', linewidth=1, markersize=8)
    plt.gca().set_aspect('equal', adjustable='box')
    if quality != 0:
        center = mesh.get_centriod()
        q1, q2 = mesh.get_quality_3()
        _quality = round(math.sqrt(q1 * q2), 2)
        plt.text(center.x * 0.8, center.y * 0.8, str(_quality), fontsize=15)
    plt.gca().set_frame_on(False)
    plt.xticks([])
    plt.yticks([])
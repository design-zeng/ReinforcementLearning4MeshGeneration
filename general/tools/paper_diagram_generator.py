import numpy as np
import matplotlib.pyplot as plt
from general.components import Vertex, Segment

def bad_cases():
    ax1 = plt.subplot(131)
    ax1.plot([1.5, 1, 2, 1.8, 1.5], [1, 2, 1, 4, 1], 'b-o')
    ax1.set_title("(a)")

    ax2 = plt.subplot(132)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [Vertex(1, 1), Vertex(2, 0.9), Vertex(2.8, 0.8), Vertex(3, 1.9),
                                               Vertex(2.2, 2), Vertex(3.5, 1.2), Vertex(2.2, 2.2), Vertex(3.6, 2.3),
                                               Vertex(4.5, 1.3), Vertex(5.5, 1.4)]

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
        s.show()
    ax2.set_title("(b)")

    ax3 = plt.subplot(133)
    p1, p2, p3, p4, p5, p6, p7, p8 = [Vertex(1, 1), Vertex(2, 0.9), Vertex(2.8, 0.8), Vertex(2.8, 1.6),
                                               Vertex(2.2, 1.7), Vertex(3.5, 1.2), Vertex(2.2, 2.2), Vertex(3.2, 2.3),]

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
        s.show()
    ax3.set_title("(c)")

    plt.show()
# env.boundary.show()

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

    [a.get_xaxis().set_visible(False) for a in fig.axes]
    [a.get_yaxis().set_visible(False) for a in fig.axes]
    plt.show()

# primitive_rules()
def generation_partial_boundary():
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(-1, 6.5)
    ax2.set_ylim(-.5, 6)
    ax2.plot([0.3, 1.4, 1.5, 2.5, 3.5, 5.1, 3.4, 3.5, 2.2, 1, 0.5, -0.6, 0.3], [3.4, 2.6, 1.5, 1.5, 0.5, 1.4, 2.9, 4.3, 5.45, 5.7, 4.6, 4, 3.4], 'k-o')
    circle2 = plt.Circle((1.5, 1.5), 4, color='black', linestyle='--', fill=False)
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

generation_partial_boundary()
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

# coordinate_system()

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

    [a.get_xaxis().set_visible(False) for a in fig.axes]
    [a.get_yaxis().set_visible(False) for a in fig.axes]
    plt.show()

# output_types()
def read_img():
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    im1 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\6.png")
    plt.imshow(im1)
    ax1.set_title("(a) Original boundary")

    ax2 = fig.add_subplot(232)
    im2 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\1.png")
    plt.imshow(im2)
    ax2.set_title("(b) Sample 1")

    ax3 = fig.add_subplot(233)
    im3 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\12.png")
    plt.imshow(im3)
    ax3.set_title("(c) Sample 2")

    ax4 = fig.add_subplot(234)
    im4 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\19.png")
    plt.imshow(im4)
    ax4.set_title("(d) Sample 3")

    ax5 = fig.add_subplot(235)
    im5 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\22.png")
    plt.imshow(im5)
    ax5.set_title("(e) Sample 4")

    ax6 = fig.add_subplot(236)
    im6 = plt.imread("D:\python projects\meshgeneration\\rl\plots\\625-1513-smoothed\\34.png")
    plt.imshow(im6)
    ax6.set_title("(f) Sample 5")

    [a.get_xaxis().set_visible(False) for a in fig.axes]
    [a.get_yaxis().set_visible(False) for a in fig.axes]
    plt.savefig('e.png', dpi=1000)

# read_img()

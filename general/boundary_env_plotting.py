import matplotlib.pyplot as plt

from general.component_plotting import show_boundary


def render_boundary(boundary):
    plt.clf()
    show_boundary(boundary, style='b-', show=False)
    plt.pause(0.001)


def close_render():
    plt.close('all')

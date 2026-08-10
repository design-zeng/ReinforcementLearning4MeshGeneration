import matplotlib.pyplot as plt

from general.component_plotting import show_boundary


def render_boundary(boundary):
    """Live view of the boundary/mesh as it is generated (gym ``render`` hook)."""
    plt.clf()
    show_boundary(boundary, style='b-', show=False)
    plt.pause(0.001)


def close_render():
    plt.close('all')

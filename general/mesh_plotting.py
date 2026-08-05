import matplotlib.pyplot as plt

from general.component_plotting import plot_boundary


def generate_meshes_canvas(mesh_gen, meshes, quality, indexing, type, style):
    plot_boundary(mesh_gen.boundary, style=style, linewidth=1)
    for id, m in enumerate(meshes):
        center = m.get_centriod(diff=True)
        if quality and indexing:
            _quality = round(mesh_gen.get_quality(element=m, index=type), 4)
            plt.text(center.x, center.y, f"{id}; {_quality}", fontsize=6)
        elif quality:
            _quality = round(mesh_gen.get_quality(element=m, index=type), 4)
            plt.text(center.x, center.y, str(_quality), fontsize=6)
        elif indexing:
            plt.text(center.x, center.y, str(id), fontsize=4)


def save_meshes(mesh_gen, name, meshes, quality=False, indexing=False, type=0, dpi=300, style='k.-'):
    plt.clf()
    generate_meshes_canvas(mesh_gen, meshes, quality, indexing, type, style=style)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=-0, hspace=0, wspace=0)
    plt.savefig(name, dpi=dpi)
    plt.close('all')

import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import pandas as pd
from scipy.spatial import ConvexHull

from general.components import Mesh, Vertex, Boundary2D
from general.mesh import MeshGeneration
from general.polygon_reader import read_polygon
from general.boundary_env import BoudaryEnv

base_path = Path(__file__).parent.parent.parent
domains_path = base_path / "samples" / "domains"
output_path = base_path / "general" / "output"
# meshes are produced by sac/infer.py evaluation() (save_fig=True)
root = base_path / "sac" / "output" / "evaluation"

def clockwise_element(vertices):
    flatten_vs = [[v.x, v.y] for v in vertices]
    hull_vs = ConvexHull(flatten_vs)
    if len(hull_vs.vertices) == len(vertices):
        return [vertices[i] for i in reversed(hull_vs.vertices)]
    elif len(hull_vs.vertices) == 3:
        res = [vertices[i] for i in reversed(hull_vs.vertices)]
        res.extend([v for v in vertices if v not in res])
        return res
    else:
        raise ValueError('Not enough vertices!')

def generate_mesh_from_inp(filename):
    m = meshio.Mesh.read(filename, "abaqus")
    vertices = [Vertex(p[0], p[1]) for p in m.points]
    elements = []
    existing_eles = []
    for ele in m.cells_dict['quad']:
        vs = sorted(ele)
        if vs not in existing_eles:
            element = Mesh(clockwise_element([vertices[i] for i in ele]))
            element.connect_vertices()
            elements.append(element)
            existing_eles.append(vs)
    return vertices, elements, len(m.cells_dict['triangle']) if 'triangle' in m.cells_dict else 0

def discover_meshes():
    meshes = sorted(root.rglob("*.inp"))
    if not meshes:
        raise FileNotFoundError(
            f"No .inp meshes under {root}. Run `python -m sac.infer` (save_fig=True) first.")
    return meshes


def calculate_metrics(vertices, elements, metrics, metrics_ind):
    if "Element quality" in metrics_ind:
        element_qualities = []
        stretch, s_jabobian, taper = [], [], []
        min_angles, max_angles = [], []

        for ele in elements:
            # q1, q2 = ele.get_quality_3()
            element_qualities.append(ele.get_quality(type='robust')) #ele.get_quality()
            if 'Stretch' in metrics_ind:
                stretch.append(ele.get_quality(type='stretch'))
            if 'Taper' in metrics_ind:
                taper.append(ele.get_quality(type='taper'))
            if 'Scaled Jacobian' in metrics_ind:
                s_jabobian.append(ele.get_quality(type='s_jacobian'))
            angles = ele.inner_angles()
            if 'MinAngle' in metrics_ind:
                min_angles.append(min(angles))
            if 'MaxAngle' in metrics_ind:
                max_angles.append(max(angles))

        # m = sum(element_qualities) / len(element_qualities)
        # metrics['ave_elem_quality'].append([m,
        #                                     math.sqrt(sum([(_v - m) ** 2 for _v in element_qualities]) / len(element_qualities))])
        metrics['Element quality'].extend(element_qualities)
        if len(stretch):
            # m = sum(stretch) / len(stretch)
            # metrics['stretch'].append([m, math.sqrt(sum([(_v - m) ** 2 for _v in stretch]) / len(
            #     stretch))])
            metrics['Stretch'].extend(stretch)
        if len(taper):
            # m = sum(taper) / len(taper)
            # metrics['taper'].append([m, math.sqrt(sum([(_v - m) ** 2 for _v in taper]) / len(
            #     taper))])
            metrics['Taper'].extend(taper)
        if len(s_jabobian):
            # m = sum(s_jabobian) / len(s_jabobian)
            # metrics['s_jabobian'].append([m, math.sqrt(sum([(_v - m) ** 2 for _v in s_jabobian]) / len(
            #     s_jabobian))])
            metrics['Scaled Jacobian'].extend(s_jabobian)

        if len(min_angles):
            # m = sum(min_angles) / len(min_angles)
            # metrics['Minimal Angle'].append([m, math.sqrt(sum([(_v - m) ** 2 for _v in min_angles]) / len(
            #     min_angles))])
            metrics['MinAngle'].extend(min_angles)
        if len(max_angles):
            # m = sum(max_angles) / len(max_angles)
            # metrics['Maximum Angle'].append([m, math.sqrt(sum([(_v - m) ** 2 for _v in max_angles]) / len(
            #     max_angles))])
            metrics['MaxAngle'].extend(max_angles)

    if 'Singularity' in metrics_ind:
        singularity_count = [0 for i in range(len(vertices))]
        for id, v in enumerate(vertices):
            for ele in elements:
                if v in ele.vertices:
                    singularity_count[id] += 1
        metrics['Singularity'].append(sum([1 for re in singularity_count if re != 4 and re != 2 and re != 1]))
                                 # len(vertices), \
                                 # sum([1 for re in singularity_count if re != 4 and re != 2 and re != 1]), \
                                 # sum([1 for re in singularity_count if re != 4 and re != 2 and re != 1]) / len(vertices)
    if 'num_vertices' in metrics_ind:
        metrics['num_vertices'].append(len(vertices))

    if 'num_elements' in metrics_ind:
        metrics['num_elements'].append(len(elements))

    return metrics


def metrics_4_domains():
    domains = {'BQ': [], 'Pave': [], 'DG': []}
    for m in discover_meshes():
        stem = str(m.relative_to(root))[:-4]
        if stem.startswith('g_'):
            domains['BQ'].append(stem)
        elif stem.startswith('pave'):
            domains['Pave'].append(stem)
        else:
            domains['DG'].append(stem)
    metrics = {
        k: {"Element quality": [], 'Singularity': []} for k in domains
    }

    for k, v in domains.items():
        for rr in v:
            final_metrics(rr, metrics[k])

    for name, m in metrics.items():
        print(f"Method {name}")
        for k, v in m.items():
            if len(v):
                if not isinstance(v[0], list):
                    m = sum(v) / len(v)
                    print(k, m, math.sqrt(sum([(_v - m) ** 2 for _v in v]) / len(v)))
                else:
                    print(k, sum([_v[0] for _v in v]) / len(v), sum([_v[1] for _v in v]) / len(v))

    metrics_data = {
        "Element quality": pd.DataFrame({"value": [], "method": []}),
        'Singularity': pd.DataFrame({"value": [], "method": []}),
    }
    for name, m in metrics.items():
        for k, v in m.items():
            if name == 'Pave' and k == 'Triangles':
                r = {"value": [10, 4, 10], "method": [name] * len(v) if name is not None else []}
            else:
                r = {"value": v, "method": [name] * len(v) if name is not None else []}
            r = pd.DataFrame(r)
            metrics_data[k] = pd.concat([metrics_data[k], r])

    print(metrics_data)

def final_metrics(domain, metrics):
    filename = f"{root}/{domain}.inp"
    vertices, elements, tri_elements = generate_mesh_from_inp(filename)

    duplicated_vertices = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if vertices[i] is not vertices[j]:
                if vertices[i].distance_to(vertices[j]) == 0:
                    # duplicated_vertices.append((i, j))
                    duplicated_vertices.append(vertices[j])

    real_vertices = [v for v in vertices if v not in duplicated_vertices]
    du_elements = []
    for ele in elements:
        for v in ele.vertices:
            if v not in real_vertices:
                if ele not in du_elements:
                    du_elements.append(ele)
    real_elements = [ele for ele in elements if ele not in du_elements]

    if 'Triangles' in metrics.keys():
        metrics['Triangles'].append(tri_elements)
    calculate_metrics(real_vertices, real_elements, metrics, ["Element quality",  'Singularity'])

def computational_cost_a2c():
    logs = sorted((base_path / "ebrd" / "output").rglob("*rewardings*"))
    if not logs:
        raise FileNotFoundError(
            f"No A2C rewardings log under {base_path / 'ebrd' / 'output'}.")
    with open(logs[0], 'r') as fr:
        data = json.load(fr)

    num_elements = []
    num_valid_elements7 = []
    num_valid_elements5 = []
    num_valid_elements3 = []
    x = [0]
    for i, r in data['r'].items():
        x.append(int(i))
        num_elements.append(len(r))
        num_valid_elements7.append(len([q[0] for q in r if q[0] > 0.7]))
        num_valid_elements5.append(len([q[0] for q in r if q[0] > 0.5]))
        num_valid_elements3.append(len([q[0] for q in r if q[0] > 0.3]))

    # for t in data['t']:
    #     x.append(x[-1] + t)
    # x.pop(0)

    avg_num_valid_elements7 = [sum(num_valid_elements7[i - 99:i + 1])/100 for i in range(99, len(x))]
    avg_num_valid_elements5 = [sum(num_valid_elements5[i - 99:i + 1]) / 100 for i in range(99, len(x))]
    avg_num_valid_elements3 = [sum(num_valid_elements3[i - 99:i + 1]) / 100 for i in range(99, len(x))]
    avg_elements = [sum(num_elements[i - 99:i + 1]) / 100 for i in range(99, len(x))]

    plt.plot(x[99:], avg_elements, 'r-', label='All')
    plt.plot(x[99:], avg_num_valid_elements7, 'b-', label='Quality > 0.7')
    plt.plot(x[99:], avg_num_valid_elements5, 'g-', label='Quality > 0.5')
    plt.plot(x[99:], avg_num_valid_elements3, 'k-', label='Quality > 0.3')

    plt.legend(loc='upper left')
    plt.xlabel('Training time (s)')
    plt.ylabel('Num. elements')
    plt.xscale('log')
    plt.title('Average number of elements per 100 episodes')
    plt.show()

# computational_cost_a2c()
def calculate_initial_boundaries_features():
    domains = sorted(domains_path.glob("*.json"))

    envs = [BoudaryEnv(read_polygon(name)) for name in domains]
    for e in envs:
        print(len(e.all_vertices), e.boundary.get_perimeter())
        print(len(e.all_vertices) / e.boundary.get_perimeter())

# calculate_initial_boundaries_features()
def draw_elements():
    elements = [
        [[0, 0], [0, 1], [1, 1], [1, 0]],
        [[0, 0], [-0.2, 1.1], [1, 1], [1, 0]],
        [[0.2, 0.1], [-0.2, 1.1], [1.4, 1], [1, 0]],
        [[0.1, 0.1], [0, 1.4], [1, 0.5], [1, 0]],
        [[0.1, 0.2], [0, 1], [0.6, 0.6], [1, 0]],
        [[0, 0], [0.4, 1], [1, 1.1], [0.6, 0.5]],
        [[0, 0], [0.6, 1], [1, 1.1], [0.3, 0.2]],
        [[0.5, 0], [0.4, 0.3], [1, 1], [0.55, 0.06]],
        [[0.4, 0], [0.39, 0.5], [0.5, 1], [0.51, 0.52]],
        [[0, 0.5], [0.1, 0.52], [1, 0.44], [0.2, 0.48]],
    ]
    for i, ele in enumerate(elements):
        plt.subplot(2, 5, i + 1)
        Mesh([Vertex(p[0], p[1]) for p in ele]).show()
    plt.show()

# draw_elements()

def read_inp_file(filename):
    vertices, elements, _ = generate_mesh_from_inp(filename)
    boundary = Boundary2D([])
    mesh = MeshGeneration(boundary)
    mesh.generated_meshes = elements
    return mesh

def extract_samples_from_file():
    for m in discover_meshes():
        env = read_inp_file(str(m))
        samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 3, 3, index=5, radius=6, quality_threshold=0.7)
        env.save_samples(f"{output_path}/data_augmentation/{m.stem}.json",
                         {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
    print("Saved!")

if __name__ == '__main__':
    metrics = defaultdict(list)
    for m in discover_meshes():
        final_metrics(str(m.relative_to(root))[:-4], metrics)
    print({k: [round(float(x), 3) for x in v] for k, v in metrics.items()})
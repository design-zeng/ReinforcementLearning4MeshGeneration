import json, math

import matplotlib.pyplot as plt
import meshio
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull

from general.components import Mesh, Vertex, Segment, Boundary2D
from general.mesh import MeshGeneration
from sac.boundary_env import BoudaryEnv, read_polygon

base_path = 'D:\\meshingData\\ANN\\'
# root = "D:\meshingData\\baselines\logs\evaluation\\"

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
    m = meshio.Mesh.read(filename, "abaqus")  # same arguments as meshio.read
    vertices = []
    elements = []
    existing_eles = []
    for p in m.points:
        vertices.append(Vertex(p[0], p[1]))

    for ele in m.cells_dict['quad']:
        vs = sorted([ele[0], ele[1], ele[2], ele[3]])
        if vs not in existing_eles:
            element = Mesh(clockwise_element([vertices[ele[0]],
                                  vertices[ele[1]],
                                  vertices[ele[2]],
                                  vertices[ele[3]]]))
            element.connect_vertices()
            elements.append(element)
            existing_eles.append(vs)
    return vertices, elements, len(m.cells_dict['triangle']) if 'triangle' in m.cells_dict.keys() else 0


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
    domains = {
        'BQ': [
            'g_d0',
            'g_d1',
            'g_d2',
            'g_d3',
            'g_d4',
        ],
        'Pave': [
            'pave_d0',
            'pave_d1',
            'pave_d2',
            'pave_d3',
            'pave_d4',
        ],
        'DG': [
            'd0',
            'd1',
            'd2',
            'd3',
            'd4',
        ]
    }
    metrics = {
        k: {
            "Element quality": [],
            'Singularity': [],
        } for k in domains.keys()
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
                    print(k, m, math.sqrt(sum([(_v - m) ** 2 for _v in v]) / len(
                                                            v)))
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
    filename = f"{root}\\{domain}.inp"
    vertices, elements, tri_elements = generate_mesh_from_inp(filename)

    duplicated_vertices = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if vertices[i] is not vertices[j]:
                if vertices[i].distance_to(vertices[j]) == 0:
                    # duplicated_vertices.append((i, j))
                    duplicated_vertices.append(vertices[j])

        # print(sorted(dists))
    # print(duplicated_vertices)
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
    calculate_metrics(real_vertices, real_elements, metrics,
                      ["Element quality",  'Singularity' #'num_elements' #'Stretch', , #'num_vertices', , 'Taper', 'Scaled Jacobian', 'MinAngle', 'MaxAngle'
                       ])


def computational_cost_a2c(filename):
    with open(filename, 'r') as fr:
        data = json.load(fr)
        fr.close()

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

    avg = plt.plot(x[99:], avg_elements, 'r-', label='All')
    avg7 = plt.plot(x[99:], avg_num_valid_elements7, 'b-', label='Quality > 0.7')
    avg5 = plt.plot(x[99:], avg_num_valid_elements5,
                   'g-', label='Quality > 0.5')
    avg3 = plt.plot(x[99:],
                   avg_num_valid_elements3, 'k-', label='Quality > 0.3')

    plt.legend(loc='upper left')
    plt.xlabel('Training time (s)')
    plt.ylabel('Num. elements')
    plt.xscale('log')
    plt.title('Average number of elements per 100 episodes')
    plt.show()

# computational_cost_a2c("D:\\meshingData\\A2C\\plots\\test_62\\rewardings.txt")
def calculate_initial_boundaries_features():
    domains = []
    domains.append(f'D:/python_projects/meshgeneration/ui/domains/boundary12.json')

    envs = [BoudaryEnv(read_polygon(name)) for name in domains]
    for e in envs:
        print(len(e.all_vertices), e.boundary.get_perimeter())
        print(len(e.all_vertices) / e.boundary.get_perimeter())

# calculate_initial_boundaries_features()
def draw_elements():
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(x, y)
    # axs[0, 0].set_title("main")
    # axs[1, 0].plot(x, y ** 2)
    # axs[1, 0].set_title("shares x with main")
    # axs[1, 0].sharex(axs[0, 0])
    # axs[0, 1].plot(x + 1, y + 1)
    # axs[0, 1].set_title("unrelated")
    # axs[1, 1].plot(x + 2, y + 2)
    # axs[1, 1].set_title("also unrelated")
    # fig.tight_layout()

    plt.subplot(2-1, 5, 1)
    ele1 = [[0, 0], [0, 1], [1, 1], [1, 0]]
    e_1 = Mesh([Vertex(p[0], p[1]) for p in ele1])
    e_1.show()

    # plt.subplot(2, 5, 2)
    # ele2 = [[0, 0], [-0.2, 1.1], [1, 1], [1, 0]]
    # e_2 = Mesh([Vertex(p[0], p[1]) for p in ele2])
    # e_2.show()

    plt.subplot(2-1, 5, 3-1)
    ele3 = [[0.2, 0.1], [-0.2, 1.1], [1.4, 1], [1, 0]]
    e_3 = Mesh([Vertex(p[0], p[1]) for p in ele3])
    e_3.show()

    # plt.subplot(2, 5, 4)
    # ele4 = [[0.1, 0.1], [0, 1.4], [1, 0.5], [1, 0]]
    # e_4 = Mesh([Vertex(p[0], p[1]) for p in ele4])
    # e_4.show()

    plt.subplot(2-1, 5, 5-2)
    ele5 = [[0.1, 0.2], [0, 1], [0.6, 0.6], [1, 0]]
    e_5 = Mesh([Vertex(p[0], p[1]) for p in ele5])
    e_5.show()

    # plt.subplot(2, 5, 6)
    # ele6 = [[0, 0], [0.4, 1], [1, 1.1], [0.6, 0.5]]
    # e_6 = Mesh([Vertex(p[0], p[1]) for p in ele6])
    # e_6.show()

    plt.subplot(2-1, 5, 7-3)
    ele7 = [[0, 0], [0.6, 1], [1, 1.1], [0.3, 0.2]]
    e_7 = Mesh([Vertex(p[0], p[1]) for p in ele7])
    e_7.show()

    # plt.subplot(2, 5, 8)
    # ele8 = [[0.5, 0], [0.4, 0.3], [1, 1], [0.55, 0.06]]
    # e_8 = Mesh([Vertex(p[0], p[1]) for p in ele8])
    # e_8.show()

    plt.subplot(2-1, 5, 9-4)
    ele9 = [[0.4, 0], [0.39, 0.5], [0.5, 1], [0.51, 0.52]]
    e_9 = Mesh([Vertex(p[0], p[1]) for p in ele9])
    e_9.show()

    # plt.subplot(2, 5, 10)
    # ele0 = [[0, 0.5], [0.1, 0.52], [1, 0.44], [0.2, 0.48]]
    # e_0 = Mesh([Vertex(p[0], p[1]) for p in ele0])
    # e_0.show()

    plt.show()

# draw_elements()

def read_inp_file(filename):
    vertices, elements, _ = generate_mesh_from_inp(filename)
    boundary = Boundary2D([])
    mesh = MeshGeneration(boundary)
    mesh.generated_meshes = elements
    return mesh

def extract_samples_from_file(filename):
    env = read_inp_file(filename)
    samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 3, 3, index=5, radius=6, quality_threshold=0.7)
    env.save_samples("D:\\meshingData\ANN\data_augmentation\\1\data_aug.json",
                     {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
    print("Saved!")

if __name__ == '__main__':
    # metrics_4_domains()
    extract_samples_from_file("D:\\g_d1.inp")
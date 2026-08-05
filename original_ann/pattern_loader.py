import numpy as np

from general.lin_alg import transformation


def get_patterns(filename):
    pattern_inputs = []
    pattern_outputs = []
    pattern_types = []
    with open(filename, 'r+') as f:
        for line in f:
            if not line.startswith("%"):
                line_data = [float(r) for r in line.split()]
                pattern_inputs.append(line_data[2 : 12])
                pattern_types.append([line_data[14]])
                pattern_outputs.append(line_data[15 : 17])

    return np.array(pattern_inputs), np.array(pattern_types), np.array(pattern_outputs)


def data_transformation(data):
    data_transformed = []
    for line in data:
        p0 = np.array([line[4], line[5]])
        p1 = np.array([line[6], line[7]])
        data_transformed.append(transformation(line, np.linalg.norm(p0 - p1), p0, p1))
    return np.array(data_transformed)

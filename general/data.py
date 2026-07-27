import numpy as np

def get_patterns(filename):
    pattern_inputs = []
    pattern_outputs = []
    pattern_types = []
    with open(filename, 'r+') as fr:
        for line in fr:
            if not line.startswith("%"):
                line_dat = [float(r) for r in line.split()]
                pattern_inputs.append(line_dat[2:12])
                pattern_types.append([line_dat[14]])
                pattern_outputs.append(line_dat[15:17])

    return np.asarray(pattern_inputs), np.asarray(pattern_types), np.asarray(pattern_outputs)

def data_transformation(data, ind_x_1, ind_y_1, ind_x0, ind_y0, ind_x1, ind_y1):
    #transformation of input and output data
    data_transf = []
    line_len = len(data[0])
    for line in data:
        mat = matrix_ops(line)
        data_transf.append(transformation(mat,
                                          np.asarray([line[ind_x_1], line[ind_y_1]]),
                        np.asarray([line[ind_x0], line[ind_y0]]),
                       np.asarray([line[ind_x1], line[ind_y1]])))

    return np.asarray(data_transf)


def transformation(matrix, dist, p0, p1):
    matrix -= p0

    # dist = np.linalg.norm(p0 - p1)
    # print(dist)
    matrix = np.divide(matrix, dist)

    theta = np.math.atan2((p1-p0)[1], (p1-p0)[0])

    rotation_matrix = np.asmatrix([
        [np.cos(theta), np.sin(theta)],
        [- np.sin(theta), np.cos(theta)]
    ])
    # print(rotation_matrix)
    # print(matrix.T)
    matrix = np.matmul(rotation_matrix, matrix.T).T

    # print(theta)
    # print(np.asarray(matrix))
    return np.asarray(matrix).reshape(-1)


def matrix_ops(arra):
    arra = np.asarray(arra, dtype=float)
    matrix = np.split(arra, len(arra)/2)
    matrix = np.asmatrix(matrix)
    return matrix


def data_detransformation(data):
    pass


def detransformation(point, dist, p0, p1):
    # dist = np.linalg.norm(p0 - p1)
    theta = 2 * np.math.pi - np.math.atan2((p1 - p0)[1], (p1 - p0)[0])
    original_point = np.empty(2)

    # remove rotation
    original_point[0] = np.cos(theta) * point[0] + np.sin(theta) * point[1]
    original_point[1] = -np.sin(theta) * point[0] + np.cos(theta) * point[1]

    #remove scaling
    original_point *= dist

    #remove translation
    original_point[0] += p0[0]
    original_point[1] += p0[1]

    return original_point

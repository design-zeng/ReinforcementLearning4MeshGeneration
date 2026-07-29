import numpy as np

from general.lin_alg import transformation

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
    for line in data:
        data_transf.append(transformation(line,
                                          np.asarray([line[ind_x_1], line[ind_y_1]]),
                        np.asarray([line[ind_x0], line[ind_y0]]),
                       np.asarray([line[ind_x1], line[ind_y1]])))

    return np.asarray(data_transf)

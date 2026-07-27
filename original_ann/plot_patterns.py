import matplotlib.pyplot as plt
import numpy as np

from general import data

def plot_patterns():
    inputs, output_types, outputs = data.get_patterns("pattern.txt")
    # transfered_data = data.data_transformation(np.concatenate((inputs, outputs), axis=1), 4, 5, 6, 7)

    points = inputs
    for i, point in enumerate(points):
        if i < 100:
            continue
        # plot existing points
        input_x = [point[i] for i in range(len(point)) if i % 2 == 0]
        input_y = [point[i] for i in range(len(point)) if i % 2 == 1]
        # input_x.append(input_x[0])
        # input_y.append(input_y[0])
        plt.plot(input_x, input_y, 'r-')

        #plot generated point and lines
        new_x = [point[2], outputs[i][0], point[6]]
        new_y = [point[3], outputs[i][1], point[7]]

        plt.plot(new_x, new_y, 'b-')

        plt.show()
        plt.close()

# plot_patterns()
def plot_flat_points(point):
    input_x = [point[i] for i in range(len(point)) if i % 2 == 0]
    input_y = [point[i] for i in range(len(point)) if i % 2 == 1]
    # input_x.append(input_x[0])
    # input_y.append(input_y[0])
    plt.plot(input_x, input_y, 'ro-')
    plt.show()
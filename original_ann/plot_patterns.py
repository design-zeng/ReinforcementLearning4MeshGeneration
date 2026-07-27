from pathlib import Path

import matplotlib.pyplot as plt

from general import data

root_path = Path(__file__).parent.parent
pattern_path = root_path / "original_ann" / "patterns" / "pattern.txt"

def plot_patterns():
    inputs, output_types, outputs = data.get_patterns(pattern_path)

    points = inputs
    for i, point in enumerate(points):
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

def plot_flat_points(point):
    input_x = [point[i] for i in range(len(point)) if i % 2 == 0]
    input_y = [point[i] for i in range(len(point)) if i % 2 == 1]
    # input_x.append(input_x[0])
    # input_y.append(input_y[0])
    plt.plot(input_x, input_y, 'ro-')
    plt.show()

if __name__ == "__main__":
    plot_patterns()
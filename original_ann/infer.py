import numpy as np
import torch

from general.lin_alg import detransformation
from general.components import Vertex
from original_ann.pattern_loader import get_patterns, data_transformation
from original_ann.model import load_model, pattern_path, device


def predict(model, points):
    model.eval()

    flattened_points = []
    for point in points:
        flattened_points.append(point.x)
        flattened_points.append(point.y)

    transformed_data = data_transformation([flattened_points], 2, 3, 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transformed_data[:, : 4], transformed_data[:, -4: -2]), axis=1)).float().to(device)

    with torch.no_grad():
        predict = model.forward(x)

    p0 = np.array([flattened_points[4], flattened_points[5]])
    p1 = np.array([flattened_points[6], flattened_points[7]])
    distance = np.linalg.norm(p0 - p1)  # base length used to scale during transformation
    detransformed_predict = detransformation(
        np.array([predict[0][1], predict[0][2]]),
        distance,
        p0,
        p1)
    return round(float(predict[0][0])), detransformed_predict


if __name__ == "__main__":
    # Load the trained model (run `python -m original_ann.train` first) and predict the
    # next element on a boundary configuration taken from the shipped pattern set.
    model = load_model()
    inputs, _, _ = get_patterns(pattern_path)
    coords = inputs[0]  # a local configuration: 5 (x, y) boundary points
    pts = [Vertex(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
    element_type, new_vertex = predict(model, pts)
    print(f"element type={element_type}, new vertex={new_vertex}")

import numpy as np
import torch

from general.geometry import detransformation
from general.geometry import Vertex
from original_ann.pattern_loader import get_patterns, data_transformation
from original_ann.model import load_model, pattern_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, vertices):
    model.eval()

    flattened_vertices = []
    for vertex in vertices:
        flattened_vertices.append(vertex.x)
        flattened_vertices.append(vertex.y)

    transformed_data = data_transformation([flattened_vertices])

    x = np.concatenate((transformed_data[:, : 4], transformed_data[:, -4: -2]), axis=1)
    x = torch.from_numpy(x).float().to(device)

    with torch.no_grad():
        y = model.forward(x)

    vertex_to_detransform = np.array([y[0][1], y[0][2]])
    p0 = np.array([flattened_vertices[4], flattened_vertices[5]])
    p1 = np.array([flattened_vertices[6], flattened_vertices[7]])
    detransformed_predict = detransformation(vertex_to_detransform, np.linalg.norm(p0 - p1), p0, p1)

    action_type = round(float(y[0][0]))

    return action_type, detransformed_predict


if __name__ == "__main__":
    model = load_model()
    input_points, _, _ = get_patterns(pattern_path)
    input_vertices = [Vertex(input_points[0][i], input_points[0][i + 1]) for i in range(0, 10, 2)]
    action_type, new_vertex = predict(model, input_vertices)
    print(f"action type={action_type}, new vertex={new_vertex}")

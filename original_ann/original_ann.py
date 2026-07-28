from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from general import data

root_path = Path(__file__).parent.parent
model_path = root_path / "original_ann" / "output" / "model.pt"
pattern_path = root_path / "original_ann" / "patterns" / "pattern.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-4
epoches = 300000
D_in, H1, H2, D_out = 6, 500, 500, 3

def new_model():
    model = nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        # nn.Tanh(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        # nn.Tanh(),
        nn.Linear(H2, D_out),
    ).to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    return model

def load_model():
    model = new_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def build_training_data():
    inputs, output_types, outputs = data.get_patterns(pattern_path)
    transformed_data = data.data_transformation(np.concatenate((inputs, outputs), axis=1), 2, 3, 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transformed_data[:, : 4], transformed_data[:, -4: -2]), axis=1)).float()
    y = torch.from_numpy(np.concatenate((output_types, transformed_data[:, -2:]), axis=1)).float()

    x.to(device)
    y.to(device)

    return x, y

def train(model, x, y):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epoches):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if loss < 0.03:
            break
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)

def predict(model, points):
    model.eval()

    flattened_points = []
    for point in points:
        flattened_points.append(point.x)
        flattened_points.append(point.y)

    transformed_data = data.data_transformation([flattened_points], 2, 3, 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transformed_data[:, : 4], transformed_data[:, -4: -2]), axis=1)).float().to(device)

    with torch.no_grad():
        predict = model.forward(x)

    p0 = np.array([flattened_points[4], flattened_points[5]])
    p1 = np.array([flattened_points[6], flattened_points[7]])
    distance = np.linalg.norm(p0 - p1)  # base length used to scale during transformation
    detransformed_predict = data.detransformation(
        np.array([predict[0][1], predict[0][2]]),
        distance,
        p0,
        p1)
    return round(float(predict[0][0])), detransformed_predict

if __name__ == "__main__":
    model = new_model()
    # model = load_model()

    x, y = build_training_data()

    train(model, x, y)

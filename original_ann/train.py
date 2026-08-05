import numpy as np
import torch
import torch.nn as nn

from original_ann.pattern_loader import get_patterns, data_transformation
from original_ann.model import new_model, load_model, model_path, pattern_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-4
EPOCHES = 300000


def build_training_data():
    input_points, output_types, output_points = get_patterns(pattern_path)

    points = np.concatenate((input_points, output_points), axis=1)
    transformed_points = data_transformation(points)

    x = np.concatenate((transformed_points[:, : 4], transformed_points[:, -4: -2]), axis=1)
    y = np.concatenate((output_types, transformed_points[:, -2:]), axis=1)

    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    return x, y


def train(model, x, y):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for t in range(EPOCHES):
        pred = model(x)

        loss = loss_fn(pred, y)
        if loss < 0.03:
            break
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    model = new_model()
    # model = load_model()

    x, y = build_training_data()

    train(model, x, y)
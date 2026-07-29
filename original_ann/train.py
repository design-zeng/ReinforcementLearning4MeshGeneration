import numpy as np
import torch
import torch.nn as nn

from original_ann.pattern_loader import get_patterns, data_transformation
from original_ann.model import new_model, load_model, model_path, pattern_path, device

learning_rate = 1e-4
epoches = 300000


def build_training_data():
    inputs, output_types, outputs = get_patterns(pattern_path)
    transformed_data = data_transformation(np.concatenate((inputs, outputs), axis=1), 2, 3, 4, 5, 6, 7)

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


if __name__ == "__main__":
    model = new_model()
    # model = load_model()  # resume from a checkpoint instead of training fresh

    x, y = build_training_data()

    train(model, x, y)

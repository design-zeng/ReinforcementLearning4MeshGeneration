import torch
import numpy as np

from general import data

model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_in, H1, H2, D_out = 6, 500, 500, 3

def new_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        # torch.nn.Tanh(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        # torch.nn.Tanh(),
        torch.nn.Linear(H2, D_out),
    ).to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    return model

def build_model():
    inputs, output_types, outputs = data.get_patterns("pattern.txt")
    x, y = build_training_data(inputs, output_types, outputs)

    model = new_model()
    x = x.to(device)
    y = y.to(device)
    return model, x, y

def build_training_data(inputs, output_types, outputs):
    transfered_data = data.data_transformation(np.concatenate((inputs, outputs), axis=1), 2, 3, 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transfered_data[:, : 4], transfered_data[:, -4: -2]), axis=1)).float()

    y = transfered_data[:, -2:]
    y = torch.from_numpy(np.concatenate((output_types, y), axis=1)).float()
    return x, y

def training(model, x, y):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    epoches = 300000

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epoches):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if loss < 0.03:
            break
        print(t, loss.item())

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    torch.save(model.state_dict(), 'model.pt')

def load_model(path):
    global model
    if not model:
        model = new_model()
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def predict(path, points):
    if len(points) != 5:
        return

    flated_points = []
    for point in points:
        flated_points.append(point.x)
        flated_points.append(point.y)

    transfered_data = data.data_transformation([flated_points], 2, 3, 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transfered_data[:, : 4], transfered_data[:, -4: -2]), axis=1)).float()

    predict = model.forward(x)
    detran_predict = data.detransformation(np.asarray([predict[0][1], predict[0][2]]),
                                           #missing distance between p0 and p1
                                           np.asarray([flated_points[4], flated_points[5]]),
                                           np.asarray([flated_points[6], flated_points[7]]))
    return round(float(predict[0][0])), detran_predict

# model, x, y = build_model()
# training(model, x, y)

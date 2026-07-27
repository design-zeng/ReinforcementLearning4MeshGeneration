import torch
from general import data
import numpy as np
import pycuda.driver as cuda
from original_ann import plot_patterns

cuda0 = torch.device('cuda:0')
torch.set_num_threads(11)
model = None

N, D_in, H1, H2, D_out = 220, 6, 500, 500, 3

def new_model(is_cuda=False):
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        # torch.nn.Tanh(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        # torch.nn.Tanh(),
        torch.nn.Linear(H2, D_out),
    )
    if is_cuda:
        model = model.to(cuda0)
        model = torch.nn.DataParallel(model)
    return model

def build_model(is_cuda=False):

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.

    # Create random Tensors to hold inputs and outputs
    # x = torch.randn(N, D_in)
    # y = torch.randn(N, D_out)

    inputs, output_types, outputs = data.get_patterns("pattern.txt")
    x, y = build_training_data(inputs, output_types, outputs)
    # y = y.to(cuda0)

    model = new_model(is_cuda)
    # model = model.cuda()
    if is_cuda:
        x = x.to(cuda0)
        y = y.to(cuda0)
    return model, x, y

def build_training_data(inputs, output_types, outputs):
    transfered_data = data.data_transformation(np.concatenate((inputs, outputs), axis=1), 4, 5, 6, 7)

    x = torch.from_numpy(np.concatenate((transfered_data[:, : 4], transfered_data[:, -4: -2]), axis=1)).float()
    # x = x.to(cuda0)

    y = transfered_data[:, -2:]
    y = torch.from_numpy(np.concatenate((output_types, y), axis=1)).float()
    return x, y

def trainning(model, x, y):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    epoches = 300000

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epoches):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if loss < 0.03:
            break
        print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    torch.save(model.state_dict(), 'model.pt')


def load_model(path, is_cuda=False):
    global model
    if not model:
        model = new_model(is_cuda)
        model.load_state_dict(torch.load(path))
        model.eval()
        if is_cuda:
            model.to(cuda0)
    return model

# model = load_model('model.pt')

def predict(path, points):
    if len(points) != 5:
        return

    flated_points = []
    for point in points:
        flated_points.append(point.x)
        flated_points.append(point.y)

    # flated_points = [0,  0,   0,  1,  0,  2,   1,  2,   2,  2]
    # flated_points = [-5.97093, -4.74350,  -6.18946, -5.36206,  -6.41700, -5.92970,  -5.92203, -5.95598,  -5.42707, -5.98226]
    # plotting.plot_flat_points(flated_points)

    transfered_data = data.data_transformation([flated_points], 4, 5, 6, 7)

    # plt.plot([transfered_data[0][i] for i in range(len(transfered_data[0])) if i % 2 == 0],
    #          [transfered_data[0][i] for i in range(len(transfered_data[0])) if i % 2 == 1], 'ro-')
    # plt.show()

    x = torch.from_numpy(np.concatenate((transfered_data[:, : 4], transfered_data[:, -4: -2]), axis=1)).float()

    predict = model.forward(x)
    # print(predict)
    # predict = [[9, 0, -2]]
    detran_predict = data.detransformation(np.asarray([predict[0][1], predict[0][2]]),
                                           np.asarray([flated_points[4], flated_points[5]]),
                                           np.asarray([flated_points[6], flated_points[7]]))
    # print(detran_predict)
    return round(float(predict[0][0])), detran_predict

# model, x, y = build_model()
# trainning(model, x, y)
# model = load_model('model.pt')
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from general.polygon_reader import read_polygon
from general.boundary_env import BoundaryEnv
from general.component_plotting import savefig_boundary
from ebrd.data_augmentation import sampling_main
from ebrd.model import FNNPolicy, get_action, device, SEED, domains_path, output_path, augmentation_path


def load_training_data(file_name):
    with open(file_name, 'r') as fr:
        return json.loads(fr.read())


def build_training_data(data):
    if len(data['samples']) == 0:
        raise ValueError(
            "No training samples to build from — mesh extraction produced none. "
            "Try a lower quality_threshold or a better-trained model.")
    inputs = np.array(data['samples'])
    outputs = np.array(data['outputs'])
    output_types = np.array(data['output_types'])

    x = torch.from_numpy(inputs).float().to(device)
    y = torch.from_numpy(np.concatenate((output_types, outputs), axis=1)).float().to(device)
    return x, y

"""
Train the FNN with a single joint MSE loss (FreeMesh-S paper, Eq. 21).

train_fnn() is the improved variant that splits the type (classification)
and coordinate (regression) objectives, and is what start_training() uses in practice.
"""
# def train_fnn_mse(model, x, y, model_path, tensorboard_log, epoches=500000):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     learning_rate = 3e-4
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     writer = SummaryWriter(tensorboard_log)
#     running_loss = []
#     for t in range(epoches):
#         y_actions, y_types = model(x)
#         y_pred = torch.cat([y_types, y_actions], 1).to(device)
#         loss = loss_fn(y_pred, y)
#         if loss < 0.01:
#             break
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss.append(loss.item())
#         if t % 1000:
#             writer.add_scalar('training loss', sum(running_loss[-1000:]) / 1000, t)
#     torch.save(model.state_dict(), model_path)


def types_to_class_indices(types):
    """Map the environment type encoding {0, 0.5, 1} to class indices {0, 1, 2}."""
    result = []
    for value in types:
        if value.item() == 0:
            result.append(0)
        elif value.item() == 0.5:
            result.append(1)
        elif value.item() == 1:
            result.append(2)
        else:
            raise ValueError("Unsupported value: {}".format(value.item()))
    return torch.tensor(result)


def train_fnn(train_data, model, num_epoches, batch_size, tensorboard_log, model_path, lr=None):
    """Train the FNN with split objectives: cross-entropy on the element type and
    MSE on the vertex coordinates (an improvement over the paper's single joint
    MSE in train_fnn_mse)."""
    torch.manual_seed(SEED)  # reproducible weight init order / DataLoader shuffling
    optimizer = optim.Adam(model.parameters(), lr=lr or 1e-3)

    classification_loss_fn = nn.CrossEntropyLoss(reduction='sum')  # element type
    regression_loss_fn = nn.MSELoss(reduction='sum')  # vertex coordinates

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(tensorboard_log)

    for t in range(num_epoches):
        size = len(train_dataloader)
        sum_loss = 0
        for batch, (x, y) in enumerate(train_dataloader):
            y_actions, y_types = model(x)

            classification_loss = classification_loss_fn(y_types, types_to_class_indices(y[:, 0]).to(device))
            regression_loss = regression_loss_fn(y_actions, y[:, 1:].to(device))
            loss = classification_loss + regression_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if batch % 10:
                writer.add_scalar('training loss', sum_loss / 1000, t * size + batch)

        print(f"loss: {sum_loss / size:>7f}  [{t:>5d}/{num_epoches:>5d}]")

    print("Done!")
    torch.save(model.state_dict(), model_path)


def start_training(model_path, data_path, tensorboard_log):
    """Load extracted samples, build tensors, and train a fresh FNN policy."""
    x, y = build_training_data(load_training_data(data_path))
    model = FNNPolicy().to(device)
    train_fnn(list(zip(x, y)), model, 2000, 128, tensorboard_log, model_path)


def self_evolving_training(env, version, model=None, episodes=100, max_steps=8000):
    """Self-evolving loop of FreeMesh-S (paper Table 6).

    Each round: mesh the domain with the current FNN policy, harvest fresh
    training samples from the good-quality elements it produced, and retrain the
    policy on those samples, so the model bootstraps itself from its own output.
    """
    model = model or FNNPolicy().to(device)
    running_reward = 10
    step = 0

    plots_dir = output_path / "plots" / version
    samples_dir = output_path / "samples" / version
    models_dir = output_path / "models" / version
    logs_dir = output_path / "log" / version
    for directory in (plots_dir, samples_dir, models_dir, logs_dir):
        os.makedirs(directory, exist_ok=True)

    for i_episode in range(episodes):
        state, ep_reward = env.reset(static=True), 0
        start = time.time()

        for _ in range(max_steps):
            step += 1
            action, type_value = get_action(state, model)
            state, reward, done, _ = env.move(action, round(type_value, 2), 0.9, 0.5)
            print(_, reward, len(env.updated_boundary.vertices))
            ep_reward += reward
            if done:
                break

        print(f"Execution time: {time.time() - start}s.")

        if len(env.updated_boundary.vertices) <= 5:
            env.smooth(env.boundary.vertices)
        else:
            env.smooth_pave(env.boundary.vertices, env.updated_boundary.vertices, iteration=400, interior=True)

        savefig_boundary(env.boundary, plots_dir / f"{i_episode}.png", style='k-', dpi=300)
        print("Figure saved!")

        running_reward = 0.05 * ep_reward + 0.95 * running_reward

        samples, output_types, outputs = env.extract_samples_2(
            env.generated_meshes, 2, 3, radius=4, quality_threshold=0.7)
        env.save_samples(samples_dir / f"ebrd_{i_episode}.json",
                         {'samples': samples, 'output_types': output_types, 'outputs': outputs},
                         _type=2)

        if not samples:
            print("No good-quality elements extracted this round; skipping retrain.")
            continue

        x, y = build_training_data({'samples': samples,
                                    'output_types': output_types,
                                    'outputs': outputs})
        train_fnn(list(zip(x, y)), model, 10000, 1024,
                  tensorboard_log=logs_dir, model_path=models_dir / f"ebrd_model_{i_episode}.pt")

        print("%d: done %d games, running reward %.3f" % (step, i_episode, running_reward))


def data_sampling(data_path, n, threshold):
    """Experience Extraction: sample training data filtered by a mesh-quality
    threshold (FreeMesh-S)."""
    os.makedirs(Path(data_path).parent, exist_ok=True)
    start_time = time.time()
    sampling_main(10, n, threshold, file_name=str(data_path))
    print('Sampling completed in', time.time() - start_time, 's!')


def hyperparameter_search():
    """Quality-threshold ablation (FreeMesh-S, Table 7): sweep the extraction
    quality threshold, then sample -> train -> evaluate for each value."""
    from ebrd.infer import evaluation  # local import: the sweep also evaluates
    quality_thresholds = [i / 100 for i in range(60, 90, 2)]
    for q in quality_thresholds:
        version = f'1_2k_{q}'

        data_path = augmentation_path / "1103" / f"training_samples_{version}.json"
        os.makedirs(data_path.parent, exist_ok=True)
        data_sampling(data_path, n=40000, threshold=q)

        start_time = time.time()
        start_training(augmentation_path / f"{version}.pt",
                       data_path,
                       tensorboard_log=augmentation_path / "log" / version)
        print('Complete training in:', time.time() - start_time, 's.')

        evaluation(augmentation_path / f"{version}.pt", version, is_render=False, indexing=True,
                   save_fig=True, save_samples=False)


if __name__ == '__main__':
    version = 'run1'
    data_path = augmentation_path / version / "training_samples.json"
    model_path = augmentation_path / f"{version}.pt"

    # FreeMesh-S (Pan et al., 2021), producer steps. The output under ebrd/output/
    # feeds ebrd.infer; only samples/domains is external.

    # 1. Experience Extraction: generate FNN training samples.
    data_sampling(data_path, n=40000, threshold=0.7)

    # 2. Train the FNN policy on the extracted samples -> writes model.pt.
    start_training(model_path, data_path, tensorboard_log=augmentation_path / "log" / version)

    # Then evaluate with:  python -m ebrd.infer

    # Alternative experiments:
    # hyperparameter_search()   # sweep the extraction quality threshold (paper Table 7)
    # self_evolving_training(BoundaryEnv(read_polygon(domains_path / "random1_1.json")), version)  # paper Table 6

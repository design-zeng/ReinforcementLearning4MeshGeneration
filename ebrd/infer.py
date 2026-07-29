import os

from general.polygon_reader import read_polygon
from general.boundary_env import BoudaryEnv
from ebrd.model import get_action, load_model, domains_path, augmentation_path


def prepare_eval_envs():
    """Test domains of varying difficulty (FreeMesh-S generalizability set)."""
    names = ["engeer", "star1", "random1_1", "tool2", "test2", "test3"]
    return [BoudaryEnv(read_polygon(domains_path / f"{name}.json")) for name in names]


def evaluation(model_path, version, is_render=False, indexing=False, save_fig=False, save_samples=False):
    """Evaluate a trained FNN policy on the test domains (generalizability)."""
    out_dir = augmentation_path / version
    os.makedirs(out_dir, exist_ok=True)

    envs = prepare_eval_envs()
    model = load_model(model_path)

    for i, env in enumerate(envs):
        print(f'Starting for model with env {i}')
        state = env.reset(static=True)

        while True:
            action, type_value = get_action(state, model)
            state, reward, done, info = env.move(action, round(type_value, 2))
            if is_render:
                env.render()
            if done:
                break

        env.close()

        if save_fig:
            if info['is_complete']:
                env.smooth(env.boundary.vertices)
                env.save_meshes(out_dir / f"ebrd_env_{i}__smoothed.png",
                                meshes=env.generated_meshes,
                                indexing=indexing, style='k-')
            else:
                env.save_meshes(out_dir / f"ebrd_env_{i}.png",
                                meshes=env.generated_meshes,
                                indexing=indexing, style='k-')
        if save_samples:
            if len(env.generated_meshes):
                samples, output_types, outputs = env.extract_samples_2(env.generated_meshes, 2, 3, radius=4)
                env.save_samples(out_dir / f"ebrd_env_{i}.json",
                                 {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
                print("Saved!")


if __name__ == '__main__':
    version = 'run1'
    model_path = augmentation_path / f"{version}.pt"

    # Consumes the model.pt produced by `python -m ebrd.train`.
    evaluation(model_path, version, is_render=False, indexing=False,
               save_fig=True, save_samples=False)

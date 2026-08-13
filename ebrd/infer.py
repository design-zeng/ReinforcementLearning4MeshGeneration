import os

from general.utils import read_polygon
from ebrd.ebrd_env import Ebrd_Env
from general.smoothing import smooth_mesh
from general.sample_extraction import extract_samples, write_samples
from general.plotting import save_meshes
from ebrd.model import get_action, load_model, domains_path, augmentation_path


def prepare_eval_envs():
    names = ["airfoil", "star1", "random1_1", "tool2", "test2", "test3"]
    return [Ebrd_Env(read_polygon(domains_path / f"{name}.json")) for name in names]


def evaluation(model_path, version, is_render=False, indexing=False, save_fig=False, save_samples=False):
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
                smooth_mesh(env.mesh, env.boundary, env.mesh.boundary.vertices)
                save_meshes(env, out_dir / f"ebrd_env_{i}__smoothed.png",
                            quads=env.mesh.generated_quads,
                            indexing=indexing, style='k-')
            else:
                save_meshes(env, out_dir / f"ebrd_env_{i}.png",
                            quads=env.mesh.generated_quads,
                            indexing=indexing, style='k-')
        if save_samples:
            if len(env.mesh.generated_quads):
                samples, output_types, outputs = extract_samples(env.mesh, env.mesh.generated_quads, 2, 3, radius=4)
                write_samples(out_dir / f"ebrd_env_{i}.json",
                                 {'samples': samples, 'output_types': output_types, 'outputs': outputs}, _type=2)
                print("Saved!")


if __name__ == '__main__':
    version = 'run1'
    model_path = augmentation_path / f"{version}.pt"

    # Consumes the model.pt produced by `python -m ebrd.train`.
    evaluation(model_path, version, is_render=False, indexing=False,
               save_fig=True, save_samples=False)

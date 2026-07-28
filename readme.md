# RL for Mesh Generation

This repository contains a Python implementation of the RL-based mesh generation algorithm presented in:

1. Pan, J., Huang, J., Wang, Y., Cheng, G., & Zeng, Y. (2021). A self-learning finite element extraction system based on reinforcement learning. *AI EDAM, 35(2)*, 180-208.
2. Pan, J., Huang, J., Cheng, G., & Zeng, Y. (2023). Reinforcement learning for automatic quadrilateral mesh generation: A soft actor–critic approach. *Neural Networks, 157*, 288-304.

which were evolved from the following:

3. Zeng, Y., & Yao, S. (2009). Understanding design activities through computer simulation. *Advanced Engineering Informatics, 23(3)*, 294-308.
4. Yao, S., Yan, B., Chen, B., & Zeng, Y. (2005). An ANN-based element extraction method for automatic mesh generation. *Expert Systems with Applications, 29(1)*, 193-206.
5. Zeng, Y., & Cheng, G. (1993). Knowledge‐Based Free Mesh Generation of Quadrilateral Elements in Two‐Dimensional Domains. *Computer‐Aided Civil and Infrastructure Engineering, 8(4)*, 259-270.

If you use this implementation in your work, please add a reference/citation to the paper:

```bibtex
@article{pan2021self,
  title={A self-learning finite element extraction system based on reinforcement learning},
  author={Pan, Jie and Huang, Jingwei and Wang, Yunli and Cheng, Gengdong and Zeng, Yong},
  journal={AI EDAM},
  volume={35},
  number={2},
  pages={180--208},
  year={2021},
  publisher={Cambridge University Press}
}

@article{pan2023reinforcement,
  title={Reinforcement learning for automatic quadrilateral mesh generation: A soft actor--critic approach},
  author={Pan, Jie and Huang, Jingwei and Cheng, Gengdong and Zeng, Yong},
  journal={Neural Networks},
  volume={157},
  pages={288--304},
  year={2023},
  publisher={Elsevier}
}
```

## Setup

**Requirements:** Python 3.10+ and the packages listed in `requirements.txt` at
the repository root (PyTorch, Stable-Baselines3, Gym, Gymnasium, Shimmy,
TensorBoard, NumPy, SciPy, pandas, seaborn, matplotlib, meshio, VTK, pygame).

```bash
# optional: an isolated environment
conda create --name mesh python=3.11
conda activate mesh

# from the repository root
pip install -r requirements.txt
```

On Intel (x86_64) macOS, install PyTorch with conda instead
(`conda install pytorch -c pytorch`); PyPI no longer ships x86_64 macOS wheels.

The code runs on current PyPI releases (last tested with torch 2.5,
stable-baselines3 2.9, gym 0.26 + gymnasium, numpy 2.2).

## Repository layout

- `general/` — shared geometry/meshing library (`components.py`, `mesh.py`,
  `data.py`, `boundary_env.py`, `polygon_generators.py`, `boundary_renderer.py`)
  plus standalone utilities in `general/tools/`. The geometry classes
  (`Vertex`, `Segment`, `Boundary2D`, `Mesh`, `PointEnvironment`, …) live in
  `general/components.py` and are used by every training pipeline.
- `ebrd/` — **FreeMesh-S** (paper 1): the supervised feed-forward network (FNN)
  element-extraction model (`EBRD.py`) and its training-sample generator
  (`data_augmentation.py`).
- `original_ann/` — the original ANN element-extraction predecessor (paper 4).
- `sac/` — **FreeMesh-RL** (paper 2): Soft Actor-Critic training/evaluation and
  the associated plotting tools.
- `sac-ebd-style/` — a work-in-progress from-scratch reimplementation of the SAC
  approach (see the last section).
- `domains/` — input domain geometries (JSON lists of boundary vertices), loaded
  with `read_polygon(...)`.
- each package's `output/` — where runs write checkpoints, samples, figures and
  TensorBoard logs.

## Running

Run **every** entrypoint as a module from the repository root. The scripts use
absolute imports, so invoking a file by path does not work:

```bash
python -m sac.train      # correct
python sac/train.py      # fails with ImportError
```

Most scripts pop up matplotlib figures via `plt.show()` and expose their settings
either as module-level variables near the top of the file or as commented toggles
in the `__main__` block — open the file and edit those before running.

### Reinforcement learning — Soft Actor-Critic (paper 2, `sac/`)

- **Train** — `python -m sac.train`
  Trains an agent and writes checkpoints + TensorBoard logs under
  `sac/output/logs/<method>/<version>/<stage>/`. Configure via the globals at the
  top of `sac/train.py`: `method_name` (`'sac'`, `'ppo'`, `'ddpg'`, `'td3'`,
  `'a2c'`), `version`, `environments` (the curriculum stages, each
  `[total_timesteps, train_env, eval_env]`), and `method_settings` (per-algorithm
  hyperparameters). Full training is long — lower `total_timesteps` in
  `environments` for a quick test.
- **Evaluate** — `python -m sac.infer`
  Meshes the test domains with a trained model and saves figures/samples to
  `sac/output/evaluation/`. Inside `evaluation()`, set `model_version`, `stage`
  and the `START`/`END` checkpoint range to point at a run produced by
  `sac.train`. In the `__main__` block you can instead enable
  `replication_evaluation()` (repeated runs → statistics) or
  `element_number_box_plot()` (element-count box plots).
- **Plot training curves** — `python -m sac.tools.return_vs_time_from_tflogs`
  Reads TensorBoard event files and plots averaged return vs. time step. Point the
  log dictionaries in `__main__` (e.g. `radius_logs`) at directories under
  `sac/output/logs/...`.
- **Rule-usage analysis** — `python -m sac.tools.rule_frequency_analysis`
  Bar chart of how often each extraction rule (type 0/1/2) is used, read from the
  `history_info` files written during evaluation. Set the paths in `geometries()`.

### Supervised FNN element extraction (paper 1, `ebrd/`)

- **Generate training data** — `python -m ebrd.data_augmentation`
  "Experience Extraction": samples quality-filtered training examples in parallel
  and writes them under `ebrd/output/data_augmentation/`. Adjust the `pool`, `N`
  and `threshold` arguments in `__main__`.
- **Train / evaluate the FNN** — `python -m ebrd.EBRD`
  The `__main__` block is the FreeMesh-S pipeline as ordered, uncommentable steps:
  (1) `data_sampling` → (2) `start_training` → (3) `evaluation`. It also offers
  `hyperparameter_search()` (quality-threshold sweep, paper Table 7) and
  `self_evolving_training(...)` (the self-learning loop, paper Table 6). Set
  `version` at the top.

### Original ANN element extraction (paper 4, `original_ann/`)

- **Train** — `python -m original_ann.original_ann`
  Trains the back-propagation MLP on the pattern set in
  `original_ann/patterns/pattern.txt` and saves to `original_ann/output/model.pt`.
  In `__main__`, switch `new_model()` for `load_model()` to resume from a
  checkpoint; `predict(model, points)` runs inference on a set of boundary points.
  Adjust `epoches` / `learning_rate` at the top.
- **Visualize the patterns** — `python -m original_ann.plot_patterns`

### Mesh quality & geometry tools (`general/tools/`)

- `python -m general.tools.vtk_quality_verdict` — VTK/Verdict mesh-quality metrics
  (min/max angle, scaled Jacobian, stretch, taper) for `.inp` meshes. Set `root`
  and the `domains` list.
- `python -m general.tools.mesh_quality_comparison` — element-quality and
  singularity metrics / cross-domain comparisons; reads `.inp` meshes from
  `general/output/evaluation/`. Choose the function to run in `__main__`.
- `python -m general.tools.json_gmsh_abaqus_unv_conversion` — mesh/geometry format
  conversion (`json_to_gmsh`, `inp_to_unv`); uncomment the desired call in
  `__main__`.
- `python -m general.tools.quad_quality_fourth_vertex_sweep` — sweeps a quad's
  fourth vertex and plots element quality vs. geometry.
- `python -m general.tools.generate_airfoil_domain` — generates and plots an
  airfoil-in-a-box domain.
- `python -m general.tools.paper_diagram_generator` — regenerates the papers'
  explanatory figures (primitive rules, coordinate system, element types, …);
  uncomment the desired figure function in `__main__`.
- `python -m general.tools.polygon_editor_ui_v2` — interactive Tkinter tool for
  drawing/editing domains (requires a display).

The quality tools expect `.inp`/`.vtk` mesh files and the plotting tools expect
TensorBoard logs / `history_info` files — produce these first by running the
training/evaluation steps above.

### Tests

```bash
python -m pytest general/meshtest.py
```

## `sac-ebd-style/` (informational — not a runnable entrypoint)

A newer, from-scratch reimplementation of the Soft Actor-Critic approach
(paper 2), written directly against the Gymnasium API with a cleaner geometry
core (`polygon.py`, `boundary.py`, `mesh.py`, `geometry_lib.py`) and a
`gym_env.py` environment. It is a work in progress ("not yet in EBD style") and
ships without a training script: `Gym_Env` can be instantiated, stepped, and is
Stable-Baselines3-compatible, but there is no packaged run command, and the
directory name contains a hyphen so it cannot be imported as a module. Treat it
as a reference for the ongoing rewrite rather than a runnable pipeline.

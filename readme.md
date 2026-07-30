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

**Requirements:** Python 3.10+ and the packages pinned in `requirements.txt`
(PyTorch, Stable-Baselines3, Gym, Gymnasium, Shimmy, TensorBoard, NumPy, SciPy,
pandas, seaborn, matplotlib, meshio, VTK, pygame).

```bash
# optional: an isolated environment
conda create --name mesh python=3.11
conda activate mesh

# from the repository root
pip install -r requirements.txt
```

On Intel (x86_64) macOS, install PyTorch with conda instead
(`conda install pytorch -c pytorch`); PyPI no longer ships x86_64 macOS wheels.
The pinned versions were validated together (torch 2.5, stable-baselines3 2.9,
gym 0.26 + gymnasium, numpy 2.2). Training uses seeds from the paper, such as `999`, for
reproducibility, 

## Repository layout

- `general/` — shared geometry/meshing library (`components.py`, `mesh.py`,
  `boundary_env.py`, `point_environment.py`, `lin_alg.py`, `polygon_reader.py`,
  `boundary_renderer.py`) plus standalone utilities in `general/tools/`. The
  geometry classes (`Vertex`, `Segment`, `Boundary2D`, `Mesh`) live in
  `general/components.py`; the RL state encoder `PointEnvironment` is in
  `general/point_environment.py`.
- `ebrd/` — **FreeMesh-S** (paper 1): the supervised feed-forward network (FNN)
  element-extraction model (`model.py`), its training/inference entrypoints
  (`train.py`, `infer.py`), and its training-sample generator
  (`data_augmentation.py`).
- `original_ann/` — the original ANN element-extraction predecessor (paper 4);
  ships its training patterns in `original_ann/patterns/`.
- `sac/` — **FreeMesh-RL** (paper 2): Soft Actor-Critic training/evaluation and
  the associated plotting tools.
- `sac_ebd/` — a work-in-progress from-scratch reimplementation (see last section).
- **`samples/domains/`** — the only inputs you provide: domain geometries (JSON
  boundary-vertex lists), loaded with `read_polygon(...)`. Everything else is
  generated.
- **`<package>/output/`** — everything the scripts produce (checkpoints, training
  samples, meshes, figures, TensorBoard logs, rule histories). Git-ignored, and
  **read back by the downstream scripts**.

## Running — a downstream pipeline

The only external input is `samples/domains/`. Everything else is *produced* by
running a script, written into a `<package>/output/` folder, and then *consumed*
by the next script. So **run producers before consumers**.

Most scripts pop up matplotlib figures via `plt.show()` and expose their settings
as module-level variables or `__main__` toggles — open the file to adjust them.

### Reinforcement learning — Soft Actor-Critic (paper 2, `sac/`)

1. **Train** *(needs: domains)*

   ```bash
   python -m sac.train
   ```

   Trains an agent and writes numbered checkpoints, `best_model`, and TensorBoard
   logs to `sac/output/logs/<method>/<version>/<stage>/`. Configure the globals at
   the top of `sac/train.py`: `method_name` (`sac`/`ppo`/`ddpg`/`td3`/`a2c`),
   `version`, `environments` (curriculum stages `[total_timesteps, train_env,
   eval_env]`), `method_settings`. Full training is long — lower `total_timesteps`
   for a quick run.
2. **Evaluate** *(needs: a `sac.train` run)*

   ```bash
   python -m sac.infer
   ```

   Auto-discovers the checkpoints from step 1 (`version`/`stage` at the top of
   `sac/infer.py` must match your train run), meshes the test domains, and
   **produces** figures, `.inp` meshes, samples (`sac/output/evaluation/<version>/`)
   and rule histories (`sac/output/experiments/<version>/`). `__main__` also offers
   `replication_evaluation()` and `element_number_box_plot()`.
3. **Plot training curves** *(needs: a `sac.train` run)*

   ```bash
   python -m sac.tools.return_vs_time_from_tflogs
   ```

   Auto-discovers TensorBoard runs under `sac/output/logs/` and plots averaged
   return vs. time step.
4. **Rule-usage analysis** *(needs: a `sac.infer` run)*

   ```bash
   python -m sac.tools.rule_frequency_analysis
   ```

   Bar chart of rule (type 0/1/2) usage from the histories in
   `sac/output/experiments/`.

### Supervised FNN element extraction (paper 1, `ebrd/`)

- **Train** *(needs: domains)*

  ```bash
  python -m ebrd.train
  ```

  The FreeMesh-S producer steps, written under `ebrd/output/`: (1) `data_sampling`
  (Experience Extraction) → training samples, (2) `start_training` → FNN policy
  (`model.pt`). Also offers `hyperparameter_search()` (quality-threshold sweep,
  paper Table 7) and `self_evolving_training(...)` (the self-learning loop, paper
  Table 6).
- **Infer** *(needs: an `ebrd.train` run)*

  ```bash
  python -m ebrd.infer
  ```

  Loads `model.pt` and meshes the test domains, writing figures/samples under
  `ebrd/output/`.
- **Sampling only**

  ```bash
  python -m ebrd.data_augmentation
  ```

  Writes a training-sample dataset to `ebrd/output/data_augmentation/` (adjust
  `pool`, `N`, `threshold`).

The shared network definition (`FNNPolicy`) and checkpoint loading live in `ebrd/model.py`.

### Original ANN element extraction (paper 4, `original_ann/`)

- **Train** *(needs: the shipped `original_ann/patterns/pattern.txt`)*

  ```bash
  python -m original_ann.train
  ```

  Trains the back-propagation MLP and saves it to `original_ann/output/model.pt`.
  In `__main__` swap `new_model()` for `load_model()` to resume from a checkpoint;
  `epoches`/`learning_rate` are at the top of `train.py`.
- **Infer** *(needs: a `original_ann.train` run)*

  ```bash
  python -m original_ann.infer
  ```

  Loads `model.pt` and runs `predict(model, points)` on a boundary configuration.
- **Visualize patterns**

  ```bash
  python -m original_ann.tools.plot_patterns
  ```

The shared network definition and checkpoint loading live in `original_ann/model.py`.

### Mesh quality & geometry tools (`general/tools/`)

- VTK/Verdict metrics (min/max angle, scaled Jacobian, stretch, taper) over the
  `.inp` meshes produced under `sac/output/evaluation/`. *(needs: a `sac.infer` run)*

  ```bash
  python -m general.tools.vtk_quality_verdict
  ```
- Element-quality and singularity metrics over the produced `.inp` meshes.
  *(needs: a `sac.infer` run)*

  ```bash
  python -m general.tools.mesh_quality_comparison
  ```
- Converts a domain to Gmsh geometry (works immediately) and, if present, a
  produced `.inp` to UNV.

  ```bash
  python -m general.tools.json_gmsh_abaqus_unv_conversion
  ```
- Self-contained; sweeps a quad's fourth vertex and plots element quality vs.
  geometry.

  ```bash
  python -m general.tools.quad_quality_fourth_vertex_sweep
  ```
- Self-contained; generates and plots an airfoil-in-a-box domain.

  ```bash
  python -m general.tools.generate_airfoil_domain
  ```
- The rule/coordinate/type figures are self-contained; `read_img()` assembles a
  panel from meshes produced by `sac.infer`. Uncomment the desired function in
  `__main__`.

  ```bash
  python -m general.tools.paper_diagram_generator
  ```
- Interactive Tkinter domain editor (requires a display).

  ```bash
  python -m general.tools.polygon_editor_ui_v2
  ```

## `sac_ebd/` (paper 2, 3, in progress)

A newer, from-scratch reimplementation of the Soft Actor-Critic approach
(paper 2), following the EBD framework detailed in paper 3, written directly against the Gymnasium API with a cleaner geometry
core. To train:
  ```bash
  cd sac_ebd
  python train.py
  ```
To run:
  ```bash
  cd sac_ebd
  python infer.py
  ```
You can change the domains and timestep per domain at the top of the training file, as well as the inference domain in the inference file.

# RL for Mesh Generation

A Python implementation of the RL-based quadrilateral mesh generation algorithms from:

1. Pan, J., Huang, J., Wang, Y., Cheng, G., & Zeng, Y. (2021). A self-learning finite element extraction system based on reinforcement learning. *AI EDAM, 35(2)*, 180-208.
2. Pan, J., Huang, J., Cheng, G., & Zeng, Y. (2023). Reinforcement learning for automatic quadrilateral mesh generation: A soft actor–critic approach. *Neural Networks, 157*, 288-304.

which evolved from:

3. Zeng, Y. (2015). Environment-Based Design (EBD): a Methodology for Transdisciplinary Design. *Journal of Integrated Design and Process Science, 19(1)*, 5-20.
4. Zeng, Y., & Yao, S. (2009). Understanding design activities through computer simulation. *Advanced Engineering Informatics, 23(3)*, 294-308.
5. Yao, S., Yan, B., Chen, B., & Zeng, Y. (2005). An ANN-based element extraction method for automatic mesh generation. *Expert Systems with Applications, 29(1)*, 193-206.
6. Zeng, Y., & Cheng, G. (1993). Knowledge‐Based Free Mesh Generation of Quadrilateral Elements in Two‐Dimensional Domains. *Computer‐Aided Civil and Infrastructure Engineering, 8(4)*, 259-270.

If you use this implementation, please cite:

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

## Use it as a library

To just mesh a 2D domain without touching the RL/training internals, use `rlmesh.py`.
It wraps the trained soft actor–critic policy behind a small API:

```python
from general.mesh_io import read_polygon
from rlmesh import Mesher

boundary = read_polygon("samples/domains/random1_1.json")
result = Mesher().mesh(boundary)
print(len(result.quads), "elements,", f"{result.coverage:.0%} area covered")

result.save("mesh.inp")
# export an Abaqus .inp file
nodes, faces = result.to_arrays()
# or plain data for your own tools
```

Run it from the repo root, since the domain path is relative and the `general` package must be importable.

Or build the boundary from your own points:

```python
from general.geometry import Vertex
from general.boundary import Boundary

boundary = Boundary([Vertex(x, y) for x, y in my_points])
boundary.connect_vertices()
result = Mesher().mesh(boundary)
```
There is a convenient boundary drawer inside `general/tools`, which you can run using
```bash
python -m general.tools.polygon_editor_ui_v2  
```
It needs tkinter, which ships with Python on macOS and Windows, but needs `python3-tk` on Linux.
They are saved inside `samples/domains`, so you must wire up the correct path.

The policy is stochastic, so `mesh()` retries a domain up to `attempts` times
(default 60) until it is fully meshed. `Mesher()`
loads the shipped SAC model by default; pass `Mesher(model_path=...)` to use
another.

> Use `Mesher(engine="sac_fast")` for the vectorized policy.
> The shipped `sac_fast` model is ~16× faster than `sac`.
> However, retrain with `python -m sac_fast.train` before relying on it, the current supplied model is wrong.

## Local Setup

**Requirements:** Python 3.10+ and the packages pinned in `requirements.txt`.

```bash
git clone https://github.com/design-zeng/ReinforcementLearning4MeshGeneration.git rl-mesh
cd rl-mesh
git switch ming-cleanup-v2
pip install -r requirements.txt
```

Optionally use a conda environment (`conda create --name rlmeshenv python=3.11`).
On Intel (x86_64) macOS, install PyTorch with conda instead
(`conda install pytorch -c pytorch`); PyPI no longer ships x86_64 macOS wheels.

## Google Cloud Training Setup

The shipped `sac` and `ebrd` models were trained on a Compute Engine
**g2-standard-8** instance (1× NVIDIA L4 GPU), using a Deep Learning VM image
that comes with CUDA and PyTorch preinstalled.

1. Create a Google Cloud project.
2. Set up billing.
3. Navigate to Compute Engine, then "VM Instances". When I built this project, I used the `g2-standard-4` instance with a single L4 GPU. Use "Capacity Advisor" to check for available GPUs in your region, then select your desired region when creating the VM.
4. In the "OS and Storage" tab, change your operating system to "Deep Learning on Linux" and allocate at least 100 GB of storage.
5. The right-side panel might warn you about insufficient quota, in which case you should click on "Request Quota Adjustment". Mine got approved instantly. 
6. Click Create and SSH into the VM.
7. Clone the repo.
```bash
git clone https://github.com/design-zeng/ReinforcementLearning4MeshGeneration.git rl-mesh
cd rl-mesh
git switch ming-cleanup-v2
```
8. Install Python, create venv, switch to venv, then install the requirements
```bash
pip install -r requirements.txt
```
9. Install tmux, create a tmux session.
10. Go to section "Training and Results" in this README for further instructions.

Note: You can detach from tmux by pressing `Ctrl + B`, release both, then `D`. You can now safely `exit` the VM, and all tasks inside tmux will keep running. You can check back after a few hours, by SSH-ing again, and attaching back to your tmux session with `tmux a`. To view GPU usage, use
```bash
watch -n 1 nvidia-smi
```
When training is done, download `ebrd/output/data_augmentation/run1.pt` and `sac/output/logs/sac/77/0/best_model.zip` using SCP, and place them in the same folder on your local machine. You can now run them locally.

MAKE SURE TO DELETE YOUR INSTANCE AFTER USAGE. THEY WILL KEEP INCURRING FEES.

## Repository Layout

- `rlmesh.py` library entry point (`Mesher`)
- `general/` shared across `ebrd/` and `sac/`
  - `config.py` constants
  - `geometry.py` geometry such as vertices, segments, polygons, quads, meshes
  - `boundary.py` the advancing front: front updates, element size range
  - `recognition.py` the agent's view of the front; reference-point selection
  - `action.py` decode a policy action into the domain (local frame); rule quads
  - `quality.py` quad and front quality metrics; quad acceptance
  - `smoothing.py` mesh and front smoothing
  - `mesh_io.py` IO utilities
  - `plotting.py` rendering / figure helpers
  - `tools/` format conversion, quality comparison/verdict, paper diagrams, polygon editor
- `ebrd/` Paper 1: FreeMesh-S
  - `data_augmentation.py` random training samples
  - `sample_extraction.py` harvested training samples
  - `tools/` sample distribution figures
- `sac/` Paper 2: FreeMesh-RL
- `sac_fast/` vectorized numpy reimplementation for faster inference
- `original_ann/` Paper 4: ebrd predecessor
  - `pattern_loader.py`
  - `patterns/` training samples
    - `pattern.txt`
- `samples/domains/` JSON boundary samples
- `*/output/` model weights, logs, figures

## Training and Results

### SAC

```bash
python -m sac.train
python -m sac.infer
python -m sac.tools.return_vs_time_from_tflogs
# plot return vs time
python -m sac.tools.rule_frequency_analysis
# plot rule usage frequency
```

### EBRD

```bash
python -m ebrd.train
python -m ebrd.infer
python -m ebrd.tools.sample_distribution_plots
# plot vertex, angle, and quality distributions of the training samples
```

The plot compares the extracted samples against the generated ones when extracted samples exist under `general/output/data_augmentation`, otherwise it plots the generated ones alone. To harvest extracted samples from meshes, call `extract_samples_from_file()` in `general.tools.mesh_quality_comparison`.

### Original ANN

```bash
python -m original_ann.train
python -m original_ann.infer
```

### SAC_fast

```bash
python -m sac_fast.train
python -m sac_fast.infer
```

### Other Tools

These read the meshes that `python -m sac.infer` writes (as `.inp` files under `sac/output/evaluation`), so run that first.

Evaluate every produced mesh with the VTK quality criteria. Prints average and standard deviation for min angle, max angle, scaled Jacobian, stretch, and taper:
```bash
python -m general.tools.vtk_quality_verdict
```

Compare mesh quality across methods. Prints the quality statistics of every produced mesh, grouped by filename prefix: `g_*` is counted as BQ, `pave*` as Pave, everything else as DG (this repo). To compare against BQ or Pave, copy their meshes into `sac/output/evaluation` as `.inp` files with those prefixes:
```bash
python -m general.tools.mesh_quality_comparison
```

Regenerate the diagrams from the papers. Running it as-is shows the problematic-geometries figure (`bad_cases`); the other figures (`primitive_rules`, `generation_partial_boundary`, `coordinate_system`, `output_types`, `read_img`, `element_quality_sweep`, `experiment_boundaries`) are listed in its `__main__`, uncomment the one you want. `read_img` assembles a panel from the sac.infer figures and saves it to `general/output`:
```bash
python -m general.tools.paper_diagram_generator
```

Convert between mesh formats: writes `samples/domains/basic1.json` as a Gmsh geometry script and converts the first produced `.inp` mesh to UNV, both into `general/output`. Edit its `__main__` to convert other files:
```bash
python -m general.tools.json_gmsh_abaqus_unv_conversion
```

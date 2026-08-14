# RL for Mesh Generation

This repository contains a Python implementation of the RL-based mesh generation algorithm presented in:

1. Pan, J., Huang, J., Wang, Y., Cheng, G., & Zeng, Y. (2021). A self-learning finite element extraction system based on reinforcement learning. *AI EDAM, 35(2)*, 180-208.
2. Pan, J., Huang, J., Cheng, G., & Zeng, Y. (2023). Reinforcement learning for automatic quadrilateral mesh generation: A soft actor–critic approach. *Neural Networks, 157*, 288-304.

which were evolved from the following:

3. Zeng, Y. (2015). Environment-Based Design (EBD): a Methodology for Transdisciplinary Design. *Journal of Integrated Design and Process Science, 19(1)*, 5-20. — the design methodology whose element-extraction rules the FreeMesh work follows.
4. Zeng, Y., & Yao, S. (2009). Understanding design activities through computer simulation. *Advanced Engineering Informatics, 23(3)*, 294-308.
5. Yao, S., Yan, B., Chen, B., & Zeng, Y. (2005). An ANN-based element extraction method for automatic mesh generation. *Expert Systems with Applications, 29(1)*, 193-206.
6. Zeng, Y., & Cheng, G. (1993). Knowledge‐Based Free Mesh Generation of Quadrilateral Elements in Two‐Dimensional Domains. *Computer‐Aided Civil and Infrastructure Engineering, 8(4)*, 259-270.

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

## Use it as a library

If you just want to mesh a 2D domain and don't need the RL/training internals,
use `rlmesh.py`. It wraps the trained soft actor–critic policy behind a small API.
You supply the domain boundary; rlmesh meshes it.

```python
from general.mesh_io import read_polygon
from rlmesh import Mesher

boundary = read_polygon("samples/domains/random1_1.json")   # -> a Boundary
result = Mesher().mesh(boundary)
print(len(result.quads), "elements,", f"{result.coverage:.0%} area covered")

result.save("mesh.inp")                          # export an Abaqus .inp file
nodes, faces = result.to_arrays()                # or plain data for your own tools
```

Build the boundary from your own points instead of reading a file:

```python
from general.geometry import Vertex
from general.boundary import Boundary

boundary = Boundary([Vertex(x, y) for x, y in my_points])
boundary.connect_vertices()
result = Mesher().mesh(boundary)
```

The policy is stochastic, so `mesh()` retries a domain up to `attempts` times
(default 60) until it is fully meshed — check `result.complete`. By default
`Mesher()` loads `sac/output/logs/sac/77/0/best_model.zip`; pass
`Mesher(model_path=...)` to use another, or train one with `python -m sac.train`.

To mesh with the vectorized `sac_fast` policy instead, pass `engine="sac_fast"`:

```python
result = Mesher(engine="sac_fast").mesh(boundary)
```

This loads `sac_fast/output/model.zip` by default and returns the same
`MeshResult`.

> **The shipped `sac_fast` model is provisional — prefer the default `sac`
> engine for real meshes.** It was trained for only 30k steps per domain
> (`python -m sac_fast.train 30000`) instead of the full 200k, because the full
> run takes ~10 h on a CPU. It is roughly 16× faster than `sac` (0.5 s vs 8.1 s
> per domain), but over a 17-domain check it left **26% of elements invalid**
> (tangled, zero-area, or non-convex) against **0.03%** for `sac`, at mean
> element quality 0.33 vs 0.62. It also produces far coarser meshes: it picks
> the two vertex-removing rules ~87% of the time, so it collapses the front in
> a few large slivers rather than building refined elements. Retrain with
> `python -m sac_fast.train` (full 200k) before relying on it.

## Setup

**Requirements:** Python 3.10+ and the packages pinned in `requirements.txt`.

First, clone the repo and switch to branch `ming-cleanup-v2`
```bash
git clone https://github.com/design-zeng/ReinforcementLearning4MeshGeneration.git
cd ReinforcementLearning4MeshGeneration
git switch ming-cleanup-v2
```
Optionally, create a conda environment
```bash
conda create --name rlmeshenv python=3.11
conda activate rlmeshenv
```
Install the requirements
```
pip install -r requirements.txt
```

On Intel (x86_64) macOS, install PyTorch with conda instead
(`conda install pytorch -c pytorch`); PyPI no longer ships x86_64 macOS wheels.

## Repository Layout

- `rlmesh.py` — library entry point (`Mesher`) for meshing a domain, see above
- `general/` shared geometry + meshing library, organized after Environment-Based Design
  (Zeng & Yao 2009): vertices are the primitive objects, quads the primitive products,
  the front the product–environment interface, and meshing the recursive resolution of
  conflicts on it
  - `config.py` — all tweakable constants in one place
  - `geometry.py` — Vertex (primitive object), Segment (vertex–vertex interaction), Polygon, Quad, Mesh (quads accumulated over a polygon), geometric constructions
  - `boundary.py` — `Boundary`, the advancing front where product meets environment: reference-point selection, rule quads, quad acceptance, front updates
  - `quality.py` — quad and front quality metrics
  - `recognition.py` — the agent's bounded view of the front around a reference point
  - `action.py` — carry a policy action back into the domain (local frame decode)
  - `smoothing.py` — resolve integration conflicts among placed vertices (relax until none remain)
  - `mesh_io.py` — read a JSON polygon into a `Boundary`; write Abaqus `.inp`
  - `plotting.py` — rendering / figure helpers
  - `tools/`
    - `json_gmsh_abaqus_unv_conversion.py`
    - `mesh_quality_comparison.py`
    - `paper_diagram_generator.py`
    - `polygon_editor_ui_v2.py`
    - `vtk_quality_verdict.py`
- `ebrd/` Paper 1: FreeMesh-S
  - `data_augmentation.py` synthetic training samples, invented in observation space
  - `sample_extraction.py` training samples harvested from finished meshes
  - `model.py` model architecture, loading, saving
  - `train.py`
  - `infer.py`
- `original_ann/` Paper 4: ebrd predecessor
  - `model.py` model architecture, loading, saving
  - `train.py`
  - `infer.py` inference and evaluation scripts
  - `pattern_loader.py` txt reader and transformation
  - `patterns/` training samples
    - `pattern.txt`
- `sac/` Paper 2: FreeMesh-RL
  - `train.py`
  - `infer.py`
  - `custom_callback.py`
- `sac_fast/` vectorized approach for faster GPU inference
  - `polygon.py`, `boundary.py`, `mesh.py`, `geometry_lib.py` — numpy reimplementation of the front and mesh
  - `gym_env.py` — gymnasium environment
  - `train.py`
  - `infer.py`
- `samples/domains/` json boundary samples
- `*/output/` model zip, logs, figures

## Usage

### SAC
1. Train
   ```bash
   python -m sac.train
   ```
2. Evaluate
   ```bash
   python -m sac.infer
   ```
3. Plot return vs time
   ```bash
   python -m sac.tools.return_vs_time_from_tflogs
   ```
4. Plot rule usage frequency
   ```bash
   python -m sac.tools.rule_frequency_analysis
   ```

### SAC (fast)
1. Train
   ```bash
   python -m sac_fast.train
   ```
2. Evaluate
   ```bash
   python -m sac_fast.infer
   ```

### EBRD

- Train
  ```bash
  python -m ebrd.train
  ```
- Infer
  ```bash
  python -m ebrd.infer
  ```
- The `train.py` script samples data automatically. If you wish ONLY for the data, run
  ```bash
  python -m ebrd.data_augmentation
  ```

### Original ANN

- Train
  ```bash
  python -m original_ann.train
  ```
- Infer
  ```bash
  python -m original_ann.infer
  ```

### Other Tools

- Evaluation using VTK criteria
  ```bash
  python -m general.tools.vtk_quality_verdict
  ```
- Compare mesh quality to BQ, Pave, and DG methods
  ```bash
  python -m general.tools.mesh_quality_comparison
  ```
- Some of the diagrams generated for the papers:
  ```bash
  python -m general.tools.paper_diagram_generator
  ```
  such as problematic geometries, primitive rules, vision space, space normalization, action space, etc
- Interactive custom boundary creator
  ```bash
  python -m general.tools.polygon_editor_ui_v2
  ```

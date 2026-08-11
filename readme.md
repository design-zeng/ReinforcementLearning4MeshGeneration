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

## Use it as a library

If you just want to mesh a 2D domain and don't need the RL/training internals,
use `rlmesh.py`. It wraps the trained soft actor–critic policy behind a small API.
You supply the domain boundary; rlmesh meshes it.

```python
from general.utils import read_polygon
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
from general.mesh import Boundary

boundary = Boundary([Vertex(x, y) for x, y in my_points])
boundary.connect_vertices()
result = Mesher().mesh(boundary)
```

The policy is stochastic, so `mesh()` retries a domain up to `attempts` times
(default 60) until it is fully meshed — check `result.complete`. By default
`Mesher()` loads `sac/output/logs/sac/77/0/best_model.zip`; pass
`Mesher(model_path=...)` to use another, or train one with `python -m sac.train`.

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
- `general/` shared geometry + meshing library
  - `geometry.py` — Vertex, Segment, Polygon, Quad, coordinate transforms, geometric constructions
  - `mesh.py` — Boundary (the advancing front) and Mesh (elements, smoothing, quality metrics, sample extraction, IO)
  - `mesh_env.py` — `MeshEnv`, the base RL environment wrapping a `Mesh`
  - `plotting.py` — rendering / figure helpers
  - `utils.py` — read a JSON polygon into a `Boundary`
  - `tools/`
    - `json_gmsh_abaqus_unv_conversion.py`
    - `mesh_quality_comparison.py`
    - `paper_diagram_generator.py`
    - `polygon_editor_ui_v2.py`
    - `vtk_quality_verdict.py`
- `ebrd/` Paper 1: FreeMesh-S
  - `data_augmentation.py` training sample generator
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
- `sac_ebd/` work-in-progress from-scratch reimplementation in EBD style
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
- Visualize `pattern.txt`
  ```bash
  python -m original_ann.tools.plot_patterns
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

"""rlmesh — quadrilateral mesh generation as a library.

For people who just want to mesh a 2D domain, without touching the RL/training
code. You supply the domain boundary; rlmesh meshes it with the trained soft
actor-critic policy (Pan et al., 2023).

    from general.mesh_io import read_polygon
    from rlmesh import Mesher

    boundary = read_polygon("samples/domains/random1_1.json")   # -> a Boundary
    result = Mesher().mesh(boundary)
    print(len(result.quads), "elements,", f"{result.coverage:.0%} area")

    result.save("mesh.inp")                        # Abaqus .inp
    nodes, faces = result.to_arrays()              # plain lists for your own tools

To build a boundary from your own points instead of a file:

    from general.geometry import Vertex
    from general.boundary import Boundary
    boundary = Boundary([Vertex(x, y) for x, y in my_points])
    boundary.connect_vertices()

Two policy engines are available: ``engine="sac"`` (default, Pan et al. 2023)
and ``engine="sac_fast"``, the vectorized reimplementation trained with
``python -m sac_fast.train``:

    result = Mesher(engine="sac_fast").mesh(boundary)
"""
from pathlib import Path

from general.geometry import Vertex, Quad
from general.boundary import Boundary
from general.mesh import Mesh
from general.smoothing import smooth_mesh
from general.mesh_io import write_inp

__all__ = ["Mesher", "MeshResult", "DEFAULT_MODEL", "DEFAULT_FAST_MODEL"]

_ROOT = Path(__file__).parent
DEFAULT_MODEL = _ROOT / "sac" / "output" / "logs" / "sac" / "77" / "0" / "best_model.zip"
DEFAULT_FAST_MODEL = _ROOT / "sac_fast" / "output" / "model.zip"


class MeshResult:
    """The mesh produced for one domain."""

    def __init__(self, mesh, complete):
        self._mesh = mesh
        self.quads = mesh.generated_quads
        self.complete = complete  # True if the domain was fully meshed

    @property
    def coverage(self):
        """Fraction of the original domain area covered by generated elements."""
        return sum(q.area() for q in self.quads) / self._mesh.original_area

    def save(self, path):
        """Write the mesh to an Abaqus ``.inp`` file."""
        write_inp(self._mesh, str(path))

    def to_arrays(self):
        """Return ``(nodes, faces)``: node coordinates as ``(x, y)`` tuples and
        each quad as a 4-tuple of node indices into ``nodes``."""
        nodes = self._mesh.vertices()
        faces = [tuple(nodes.index(v) for v in q.vertices) for q in self.quads]
        return [(n.x, n.y) for n in nodes], faces


class Mesher:
    """Loads a trained meshing policy once, then meshes any number of domains."""

    def __init__(self, model_path=None, device="cpu", engine="sac"):
        from stable_baselines3 import SAC  # heavy import, kept lazy

        if engine not in ("sac", "sac_fast"):
            raise ValueError(f"Unknown engine {engine!r}; use 'sac' or 'sac_fast'.")
        self.engine = engine
        if model_path is None:
            model_path = DEFAULT_MODEL if engine == "sac" else DEFAULT_FAST_MODEL
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Place one there, or train with "
                f"`python -m {engine}.train`."
            )
        self.model = SAC.load(str(model_path), device=device)

    def mesh(self, boundary, attempts=60, smooth=True, deterministic=False):
        """Mesh the domain given by ``boundary`` (a ``general.boundary.Boundary``).

        The policy is stochastic, so the domain is retried up to ``attempts``
        times until it is fully meshed. Returns a :class:`MeshResult`; check
        ``result.complete`` to see whether meshing finished. The supplied
        ``boundary`` is not modified, and its winding (clockwise or
        counter-clockwise) doesn't matter.
        """
        if len(boundary.vertices) <= 5:
            raise ValueError(
                "Could not start meshing this boundary. Supply a simple "
                "(non-self-intersecting) polygon with more than five vertices."
            )
        if self.engine == "sac_fast":
            return self._mesh_fast(boundary, attempts, smooth, deterministic)

        from sac.sac_env import Sac_Env  # heavy import, kept lazy

        env = Sac_Env(_as_clockwise(boundary))
        if env.reset() is None:
            raise ValueError(
                "Could not start meshing this boundary. Supply a simple "
                "(non-self-intersecting) polygon with more than five vertices."
            )
        for _ in range(attempts):
            info = _rollout(self.model, env, deterministic)
            if info["is_complete"]:
                if smooth:
                    smooth_mesh(env.mesh, env.boundary)
                return MeshResult(env.mesh, complete=True)
        return MeshResult(env.mesh, complete=False)

    def _mesh_fast(self, boundary, attempts, smooth, deterministic):
        import numpy as np
        from sac_fast.gym_env import Gym_Env  # heavy import, kept lazy

        points = np.array([[v.x, v.y] for v in boundary.vertices])
        try:
            env = Gym_Env(points)
        except Exception as e:
            raise ValueError(
                f"Could not start meshing this boundary ({e}). Supply a simple "
                f"(non-self-intersecting) polygon with more than five vertices."
            ) from e
        for _ in range(attempts):
            obs, _ = env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            if env.boundary_env.is_quad_left():
                return MeshResult(_fast_to_mesh(env, smooth=smooth), complete=True)
        return MeshResult(_fast_to_mesh(env, smooth=False), complete=False)


def _as_clockwise(boundary):
    """The mesher expects clockwise boundaries; reverse counter-clockwise input
    on a copy, so the caller's boundary is left untouched."""
    v = boundary.vertices
    n = len(v)
    twice_area = sum(v[i].x * v[(i + 1) % n].y - v[(i + 1) % n].x * v[i].y for i in range(n))
    if twice_area <= 0:  # already clockwise
        return boundary
    flipped = Boundary([Vertex(p.x, p.y) for p in reversed(v)])
    flipped.connect_vertices()
    return flipped


def _fast_to_mesh(env, smooth):
    """Rebuild a ``general.mesh.Mesh`` from a sac_fast rollout so the result
    surface (quads, coverage, save, smoothing) matches the sac engine. Like
    Sac_Env, a remaining four-vertex front becomes the closing quad.

    sac_fast winds its polygons counter-clockwise, but general/ reads corner
    angles clockwise, so every ring is reversed on the way across."""
    verts = [Vertex(float(x), float(y)) for x, y in env.mesh.vertices]
    original = Boundary(verts[:len(env.initial_boundary)][::-1])
    original.connect_vertices()
    mesh = Mesh(original)

    quad_indices = list(env.mesh.quads)
    front_indices = [int(i) for i in env.mesh.boundary_indices_mapping]
    if env.boundary_env.is_quad_left() and len(front_indices) == 4:
        quad_indices.append(tuple(front_indices))
    for indices in quad_indices:
        cell = Quad([verts[i] for i in reversed(indices)])
        cell.connect_vertices()
        mesh.generated_quads.append(cell)

    if smooth:
        front = Boundary([verts[i] for i in reversed(front_indices)])
        front.connect_vertices()
        smooth_mesh(mesh, front)
    return mesh


def _rollout(model, env, deterministic):
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        if done:
            break
    env.close()
    return info

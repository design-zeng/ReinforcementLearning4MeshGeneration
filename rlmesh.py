"""rlmesh — quadrilateral mesh generation as a library.

For people who just want to mesh a 2D domain, without touching the RL/training
code. You supply the domain boundary; rlmesh meshes it with the trained soft
actor-critic policy (Pan et al., 2023).

    from general.utils import read_polygon
    from rlmesh import Mesher

    boundary = read_polygon("samples/domains/random1_1.json")   # -> a Boundary
    result = Mesher().mesh(boundary)
    print(len(result.quads), "elements,", f"{result.coverage:.0%} area")

    result.save("mesh.inp")                        # Abaqus .inp
    nodes, faces = result.to_arrays()              # plain lists for your own tools

To build a boundary from your own points instead of a file:

    from general.geometry import Vertex
    from general.mesh import Boundary
    boundary = Boundary([Vertex(x, y) for x, y in my_points])
    boundary.connect_vertices()
"""
from pathlib import Path

from sac.gym_env import Gym_Env

__all__ = ["Mesher", "MeshResult", "DEFAULT_MODEL"]

_ROOT = Path(__file__).parent
DEFAULT_MODEL = _ROOT / "sac" / "output" / "logs" / "sac" / "77" / "0" / "best_model.zip"


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
        self._mesh.write_elements_to_file(str(path))

    def to_arrays(self):
        """Return ``(nodes, faces)``: node coordinates as ``(x, y)`` tuples and
        each quad as a 4-tuple of node indices into ``nodes``."""
        nodes = list(self._mesh.original_vertices)
        for quad in self.quads:
            nodes.extend(v for v in quad.vertices if v not in nodes)
        faces = [tuple(nodes.index(v) for v in q.vertices) for q in self.quads]
        return [(n.x, n.y) for n in nodes], faces


class Mesher:
    """Loads a trained meshing policy once, then meshes any number of domains."""

    def __init__(self, model_path=DEFAULT_MODEL, device="cpu"):
        from stable_baselines3 import SAC  # heavy import, kept lazy

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Place one there, or train with "
                f"`python -m sac.train`."
            )
        self.model = SAC.load(str(model_path), device=device)

    def mesh(self, boundary, attempts=60, smooth=True, deterministic=False):
        """Mesh the domain given by ``boundary`` (a ``general.mesh.Boundary``).

        The policy is stochastic, so the domain is retried up to ``attempts``
        times until it is fully meshed. Returns a :class:`MeshResult`; check
        ``result.complete`` to see whether meshing finished. The supplied
        ``boundary`` is not modified.
        """
        env = Gym_Env(boundary)
        for _ in range(attempts):
            info = _rollout(self.model, env, deterministic)
            if info["is_complete"]:
                if smooth:
                    env.mesh.smooth(env.mesh.boundary.vertices)
                return MeshResult(env.mesh, complete=True)
        return MeshResult(env.mesh, complete=False)


def _rollout(model, env, deterministic):
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        if done:
            break
    env.close()
    return info

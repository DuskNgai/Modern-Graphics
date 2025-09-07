from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from src.surface.surface import ParametricSurface


def export_to_obj(
    filepath: str,
    surface: ParametricSurface,
    num_u_segments: int = 50,
    num_v_segments: int = 50,
) -> None:
    vertices, faces, normals = surface.get_regular_mesh(num_u_segments, num_v_segments)
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    normals_np = -normals.numpy()

    vertices_np = np.stack([
        vertices_np[..., 1],
        -vertices_np[..., 0],
        -vertices_np[..., 2],
    ], axis=-1)
    normals_np = np.stack([
        normals_np[..., 1],
        -normals_np[..., 0],
        -normals_np[..., 2],
    ], axis=-1)

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
    )
    mesh.export(filepath)

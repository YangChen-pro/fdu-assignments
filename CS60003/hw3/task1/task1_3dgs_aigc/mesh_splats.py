"""Texture-aware OBJ surface sampling and oriented splat construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class MeshSamples:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    has_texture: bool
    texture_path: Path | None


@dataclass
class _Triangle:
    vertex: tuple[int, int, int]
    texcoord: tuple[int | None, int | None, int | None]
    normal: tuple[int | None, int | None, int | None]
    material: str | None


def apply_object_b_crystal_style(
    points: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Color-grade the textured mushroom from jade stem to violet crystal cap."""
    points = np.asarray(points, dtype=np.float32)
    base = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    if points.shape[0] == 0:
        return base
    lower, upper = np.percentile(points[:, 2], [5.0, 95.0])
    height = max(float(upper - lower), 1e-6)
    normalized_z = np.clip((points[:, 2] - lower) / height, 0.0, 1.0)
    cap_weight = np.clip((normalized_z - 0.42) / 0.38, 0.0, 1.0)[:, None]
    cap_weight = cap_weight * cap_weight * (3.0 - 2.0 * cap_weight)

    luminance = (
        0.2126 * base[:, 0] + 0.7152 * base[:, 1] + 0.0722 * base[:, 2]
    )[:, None]
    brightness = np.clip(0.45 + 0.85 * luminance, 0.35, 1.0)
    jade = brightness * np.array([0.16, 0.68, 0.34], dtype=np.float32)
    violet = brightness * np.array([0.72, 0.34, 0.96], dtype=np.float32)
    stem = 0.72 * base + 0.28 * jade
    cap = 0.38 * base + 0.62 * violet
    return np.clip((1.0 - cap_weight) * stem + cap_weight * cap, 0.0, 1.0)


def sample_textured_obj(path: Path, *, max_points: int, seed: int) -> MeshSamples:
    """Sample OBJ triangles by area and preserve UV texture colors."""
    vertices: list[list[float]] = []
    vertex_colors: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[_Triangle] = []
    material_files: list[Path] = []
    current_material: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            keyword = fields[0]
            if keyword == "mtllib":
                material_files.extend(path.parent / value for value in fields[1:])
            elif keyword == "usemtl" and len(fields) > 1:
                current_material = fields[1]
            elif keyword == "v" and len(fields) >= 4:
                values = [float(value) for value in fields[1:]]
                vertices.append(values[:3])
                vertex_colors.append(values[3:6] if len(values) >= 6 else [0.72, 0.72, 0.72])
            elif keyword == "vt" and len(fields) >= 3:
                texcoords.append([float(fields[1]), float(fields[2])])
            elif keyword == "vn" and len(fields) >= 4:
                normals.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif keyword == "f" and len(fields) >= 4:
                corners = [
                    _parse_corner(token, len(vertices), len(texcoords), len(normals))
                    for token in fields[1:]
                ]
                for index in range(1, len(corners) - 1):
                    tri = (corners[0], corners[index], corners[index + 1])
                    triangles.append(
                        _Triangle(
                            vertex=tuple(corner[0] for corner in tri),
                            texcoord=tuple(corner[1] for corner in tri),
                            normal=tuple(corner[2] for corner in tri),
                            material=current_material,
                        )
                    )

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    if not triangles:
        raise ValueError(f"OBJ has no faces: {path}")

    vertex_array = np.asarray(vertices, dtype=np.float32)
    color_array = np.clip(np.asarray(vertex_colors, dtype=np.float32), 0.0, 1.0)
    texcoord_array = np.asarray(texcoords, dtype=np.float32)
    normal_array = np.asarray(normals, dtype=np.float32)
    textures = _load_material_textures(material_files)

    face_vertices = np.asarray([triangle.vertex for triangle in triangles], dtype=np.int64)
    a = vertex_array[face_vertices[:, 0]]
    b = vertex_array[face_vertices[:, 1]]
    c = vertex_array[face_vertices[:, 2]]
    face_cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(face_cross, axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        raise ValueError(f"OBJ has no non-degenerate faces: {path}")
    probabilities = np.where(valid, areas, 0.0)
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(triangles), size=max_points, replace=True, p=probabilities)
    u = rng.random(max_points)
    v = rng.random(max_points)
    outside = u + v > 1.0
    u[outside] = 1.0 - u[outside]
    v[outside] = 1.0 - v[outside]
    w = 1.0 - u - v

    chosen_vertices = face_vertices[face_indices]
    va = vertex_array[chosen_vertices[:, 0]]
    vb = vertex_array[chosen_vertices[:, 1]]
    vc = vertex_array[chosen_vertices[:, 2]]
    points = w[:, None] * va + u[:, None] * vb + v[:, None] * vc

    ca = color_array[chosen_vertices[:, 0]]
    cb = color_array[chosen_vertices[:, 1]]
    cc = color_array[chosen_vertices[:, 2]]
    sampled_colors = w[:, None] * ca + u[:, None] * cb + v[:, None] * cc

    sampled_normals = np.zeros_like(points, dtype=np.float32)
    has_texture = False
    used_texture_paths: list[Path] = []
    for output_index, face_index in enumerate(face_indices):
        triangle = triangles[int(face_index)]
        sampled_normals[output_index] = _sample_normal(
            triangle,
            normal_array,
            face_cross[int(face_index)],
            (w[output_index], u[output_index], v[output_index]),
        )
        texture_entry = textures.get(triangle.material or "")
        if texture_entry is None or any(index is None for index in triangle.texcoord):
            continue
        texture_path, texture = texture_entry
        indices = np.asarray(triangle.texcoord, dtype=np.int64)
        uv = (
            w[output_index] * texcoord_array[indices[0]]
            + u[output_index] * texcoord_array[indices[1]]
            + v[output_index] * texcoord_array[indices[2]]
        )
        sampled_colors[output_index] = _sample_texture(texture, uv)
        has_texture = True
        used_texture_paths.append(texture_path)

    texture_path = used_texture_paths[0] if used_texture_paths else None
    return MeshSamples(
        points=points.astype(np.float32),
        colors=np.clip(sampled_colors, 0.0, 1.0).astype(np.float32),
        normals=_normalize_vectors(sampled_normals),
        has_texture=has_texture,
        texture_path=texture_path,
    )


def normal_aligned_quaternions(normals: np.ndarray) -> np.ndarray:
    """Return normalized WXYZ quaternions rotating local +Z onto normals."""
    targets = _normalize_vectors(np.asarray(normals, dtype=np.float32))
    quaternions = np.empty((targets.shape[0], 4), dtype=np.float32)
    dots = targets[:, 2]
    opposite = dots < -0.999999
    quaternions[opposite] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    regular = ~opposite
    quaternions[regular, 0] = 1.0 + dots[regular]
    quaternions[regular, 1] = -targets[regular, 1]
    quaternions[regular, 2] = targets[regular, 0]
    quaternions[regular, 3] = 0.0
    return _normalize_vectors(quaternions)


def quaternion_normals(quaternions: np.ndarray) -> np.ndarray:
    """Recover the rotated local +Z axis from WXYZ quaternions."""
    q = _normalize_vectors(np.asarray(quaternions, dtype=np.float32))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    result = np.empty((q.shape[0], 3), dtype=np.float32)
    result[:, 0] = 2.0 * (x * z + w * y)
    result[:, 1] = 2.0 * (y * z - w * x)
    result[:, 2] = 1.0 - 2.0 * (x * x + y * y)
    return _normalize_vectors(result)


def surface_splat_scales(
    *,
    point_count: int,
    tangent_scale: float | np.ndarray,
    thickness_ratio: float,
) -> np.ndarray:
    """Build XY-tangent/Z-normal anisotropic scales for surface splats."""
    tangent = np.broadcast_to(
        np.asarray(tangent_scale, dtype=np.float32),
        (point_count,),
    )
    return np.column_stack(
        [tangent, tangent, tangent * float(thickness_ratio)]
    ).astype(np.float32)


def _parse_corner(
    token: str,
    vertex_count: int,
    texcoord_count: int,
    normal_count: int,
) -> tuple[int, int | None, int | None]:
    parts = token.split("/")
    return (
        _resolve_index(parts[0], vertex_count),
        _resolve_index(parts[1], texcoord_count) if len(parts) > 1 and parts[1] else None,
        _resolve_index(parts[2], normal_count) if len(parts) > 2 and parts[2] else None,
    )


def _resolve_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def _load_material_textures(
    material_files: list[Path],
) -> dict[str, tuple[Path, np.ndarray]]:
    textures: dict[str, tuple[Path, np.ndarray]] = {}
    for material_file in material_files:
        if not material_file.exists():
            continue
        current = ""
        for raw in material_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = raw.strip().split()
            if not fields:
                continue
            if fields[0] == "newmtl" and len(fields) > 1:
                current = fields[1]
            elif fields[0] == "map_Kd" and len(fields) > 1 and current:
                texture_path = material_file.parent / " ".join(fields[1:])
                if texture_path.exists():
                    texture = np.asarray(Image.open(texture_path).convert("RGB"), dtype=np.float32) / 255.0
                    textures[current] = (texture_path, texture)
    return textures


def _sample_normal(
    triangle: _Triangle,
    normals: np.ndarray,
    face_cross: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    if normals.size and all(index is not None for index in triangle.normal):
        indices = np.asarray(triangle.normal, dtype=np.int64)
        return (
            weights[0] * normals[indices[0]]
            + weights[1] * normals[indices[1]]
            + weights[2] * normals[indices[2]]
        )
    return face_cross


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    x = int(np.clip(round(float(uv[0] % 1.0) * (width - 1)), 0, width - 1))
    y = int(np.clip(round((1.0 - float(uv[1] % 1.0)) * (height - 1)), 0, height - 1))
    return texture[y, x]


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (values / norms).astype(np.float32)

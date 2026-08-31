"""Icosahedral mesh utilities for GraphCast-style global models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriangularMesh:
    vertices: np.ndarray
    faces: np.ndarray


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    return xyz / np.maximum(norm, 1e-12)


def lat_lon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    return np.stack(
        (
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ),
        axis=-1,
    ).astype(np.float32)


def xyz_to_lat_lon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = normalize_xyz(xyz)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def base_icosahedron() -> TriangularMesh:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return TriangularMesh(normalize_xyz(vertices).astype(np.float32), faces)


def refine_mesh(mesh: TriangularMesh) -> TriangularMesh:
    vertices = mesh.vertices.astype(np.float64).tolist()
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in midpoint_cache:
            midpoint = normalize_xyz(
                (np.asarray(vertices[a]) + np.asarray(vertices[b]))[None, :]
            )[0]
            midpoint_cache[key] = len(vertices)
            vertices.append(midpoint.tolist())
        return midpoint_cache[key]

    new_faces = []
    for a, b, c in mesh.faces:
        ab = midpoint_index(int(a), int(b))
        bc = midpoint_index(int(b), int(c))
        ca = midpoint_index(int(c), int(a))
        new_faces.extend(
            (
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            )
        )
    return TriangularMesh(
        normalize_xyz(np.asarray(vertices, dtype=np.float64)).astype(np.float32),
        np.asarray(new_faces, dtype=np.int64),
    )


def get_hierarchy(refinement: int) -> list[TriangularMesh]:
    if refinement < 0:
        raise ValueError("refinement must be non-negative")
    meshes = [base_icosahedron()]
    for _ in range(refinement):
        meshes.append(refine_mesh(meshes[-1]))
    return meshes


def faces_to_bidirectional_edges(faces: np.ndarray) -> np.ndarray:
    undirected = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = int(u), int(v)
            undirected.add((u, v) if u < v else (v, u))

    directed = []
    for u, v in sorted(undirected):
        directed.append((u, v))
        directed.append((v, u))
    return np.asarray(directed, dtype=np.int64)


def get_multimesh_edges(hierarchy: list[TriangularMesh]) -> np.ndarray:
    edge_set = set()
    for mesh in hierarchy:
        edges = faces_to_bidirectional_edges(mesh.faces)
        for sender, receiver in edges:
            edge_set.add((int(sender), int(receiver)))
    return np.asarray(sorted(edge_set), dtype=np.int64)


def regular_lat_lon_grid(
    height: int,
    width: int,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(lat_min, lat_max, height, dtype=np.float32)
    lon = np.linspace(lon_min, lon_max, width, endpoint=False, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    xyz = lat_lon_to_xyz(lat_grid.reshape(-1), lon_grid.reshape(-1))
    return lat, lon, xyz


def lat_lon_grid(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build grid positions from the dataset's exact one-dimensional coordinates."""
    lat = np.asarray(latitudes, dtype=np.float32)
    lon = np.asarray(longitudes, dtype=np.float32)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("latitudes and longitudes must be one-dimensional")
    if lat.size == 0 or lon.size == 0:
        raise ValueError("latitudes and longitudes must be non-empty")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitudes and longitudes must be finite")
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    xyz = lat_lon_to_xyz(lat_grid.reshape(-1), lon_grid.reshape(-1))
    return lat, lon, xyz

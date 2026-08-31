"""Grid/mesh connectivity for the SLA GraphCast-style model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mesh import TriangularMesh, lat_lon_grid, normalize_xyz


@dataclass(frozen=True)
class GraphConnectivity:
    grid_xyz: np.ndarray
    mesh_xyz: np.ndarray
    mesh_edges: np.ndarray
    grid2mesh_edges: np.ndarray
    mesh2grid_edges: np.ndarray
    height: int
    width: int


def max_edge_length(mesh: TriangularMesh) -> float:
    edges = set()
    for a, b, c in mesh.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = int(u), int(v)
            edges.add((u, v) if u < v else (v, u))
    if not edges:
        return 0.0
    vertices = mesh.vertices
    return float(
        max(np.linalg.norm(vertices[u] - vertices[v]) for u, v in edges)
    )


def radius_grid_to_mesh_edges(
    grid_xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    radius: float,
) -> np.ndarray:
    grid_tree = cKDTree(grid_xyz)
    mesh_tree = cKDTree(mesh_xyz)

    edges = []
    covered_grid_nodes = np.zeros(grid_xyz.shape[0], dtype=bool)
    for mesh_index, grid_indices in enumerate(grid_tree.query_ball_point(mesh_xyz, radius)):
        if len(grid_indices) == 0:
            continue
        covered_grid_nodes[np.asarray(grid_indices, dtype=np.int64)] = True
        edges.extend((int(grid_index), mesh_index) for grid_index in grid_indices)

    uncovered = np.flatnonzero(~covered_grid_nodes)
    if uncovered.size:
        _, nearest_mesh = mesh_tree.query(grid_xyz[uncovered], k=1)
        edges.extend(
            (int(grid_index), int(mesh_index))
            for grid_index, mesh_index in zip(uncovered, np.asarray(nearest_mesh).reshape(-1))
        )

    if not edges:
        raise RuntimeError("grid-to-mesh connectivity produced no edges")
    return np.asarray(edges, dtype=np.int64)


def _points_in_spherical_faces(
    grid_xyz: np.ndarray,
    face_vertices: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return whether each unit-sphere point lies in its paired spherical face."""
    p = np.asarray(grid_xyz, dtype=np.float64)
    tri = np.asarray(face_vertices, dtype=np.float64)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]

    edge01 = np.cross(v0, v1)
    edge12 = np.cross(v1, v2)
    edge20 = np.cross(v2, v0)
    side01 = np.einsum("ij,ij->i", edge01, v2)
    side12 = np.einsum("ij,ij->i", edge12, v0)
    side20 = np.einsum("ij,ij->i", edge20, v1)
    point01 = np.einsum("ij,ij->i", edge01, p)
    point12 = np.einsum("ij,ij->i", edge12, p)
    point20 = np.einsum("ij,ij->i", edge20, p)
    return (
        (side01 * point01 >= -tolerance)
        & (side12 * point12 >= -tolerance)
        & (side20 * point20 >= -tolerance)
    )


def containing_face_mesh_to_grid_edges(
    grid_xyz: np.ndarray,
    mesh: TriangularMesh,
    initial_candidates: int = 8,
    chunk_size: int = 131_072,
) -> np.ndarray:
    """Connect each grid point to the vertices of its containing mesh face.

    A ray from the sphere centre exits the convex icosphere through the face
    whose normalized plane has the largest dot product with the ray. A KD-tree
    supplies nearby candidate planes, which are explicitly checked afterward.
    """
    points = normalize_xyz(np.asarray(grid_xyz, dtype=np.float64))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
    inward = offsets < 0
    normals[inward] *= -1.0
    offsets[inward] *= -1.0
    if np.any(offsets <= 0):
        raise RuntimeError("mesh contains a face whose plane does not bound the origin")

    unit_normals = normalize_xyz(normals)
    plane_vectors = normals / offsets[:, None]
    face_tree = cKDTree(unit_normals)
    selected = np.empty(points.shape[0], dtype=np.int64)

    candidate_counts = []
    k = max(1, min(int(initial_candidates), faces.shape[0]))
    while True:
        candidate_counts.append(k)
        if k == faces.shape[0] or k >= 128:
            break
        k = min(faces.shape[0], max(k * 4, 32))

    unresolved = np.arange(points.shape[0], dtype=np.int64)
    for candidate_count in candidate_counts:
        next_unresolved = []
        for start in range(0, unresolved.size, chunk_size):
            point_indices = unresolved[start : start + chunk_size]
            point_chunk = points[point_indices]
            _, candidates = face_tree.query(point_chunk, k=candidate_count)
            candidates = np.asarray(candidates, dtype=np.int64)
            if candidates.ndim == 1:
                candidates = candidates[:, None]

            scores = np.einsum(
                "nkj,nj->nk", plane_vectors[candidates], point_chunk
            )
            chosen = candidates[
                np.arange(candidates.shape[0]),
                np.argmax(scores, axis=1),
            ]
            inside = _points_in_spherical_faces(point_chunk, triangles[chosen])
            selected[point_indices[inside]] = chosen[inside]
            if np.any(~inside):
                next_unresolved.append(point_indices[~inside])

        if not next_unresolved:
            unresolved = np.empty(0, dtype=np.int64)
            break
        unresolved = np.concatenate(next_unresolved)

    if unresolved.size:
        raise RuntimeError(
            f"could not locate containing mesh faces for {unresolved.size} grid points"
        )

    mesh_indices = faces[selected].reshape(-1)
    grid_indices = np.repeat(np.arange(points.shape[0], dtype=np.int64), 3)
    return np.stack((mesh_indices, grid_indices), axis=1)


def build_connectivity(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    finest_mesh: TriangularMesh,
    mesh_edges: np.ndarray,
    radius_query_fraction_edge_length: float = 0.6,
    mesh2grid_candidate_faces: int = 8,
) -> GraphConnectivity:
    lat, lon, grid_xyz = lat_lon_grid(latitudes, longitudes)
    mesh_xyz = finest_mesh.vertices.astype(np.float32)
    radius = max_edge_length(finest_mesh) * radius_query_fraction_edge_length
    grid2mesh_edges = radius_grid_to_mesh_edges(grid_xyz, mesh_xyz, radius)
    mesh2grid_edges = containing_face_mesh_to_grid_edges(
        grid_xyz,
        finest_mesh,
        initial_candidates=mesh2grid_candidate_faces,
    )
    return GraphConnectivity(
        grid_xyz=grid_xyz.astype(np.float32),
        mesh_xyz=mesh_xyz.astype(np.float32),
        mesh_edges=mesh_edges.astype(np.int64),
        grid2mesh_edges=grid2mesh_edges.astype(np.int64),
        mesh2grid_edges=mesh2grid_edges.astype(np.int64),
        height=lat.size,
        width=lon.size,
    )

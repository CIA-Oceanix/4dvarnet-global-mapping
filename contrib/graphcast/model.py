"""GraphCast-style direct SLA reconstruction model."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .connectivity import GraphConnectivity, build_connectivity
from .layers import BipartiteInteractionNetwork, MeshInteractionNetwork, make_mlp
from .mesh import get_hierarchy, get_multimesh_edges


def _position_features(xyz: np.ndarray) -> np.ndarray:
    """GraphCast static position features: cos(latitude), sin/cos(longitude)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    cos_latitude = np.linalg.norm(xyz[:, :2], axis=1)
    longitude = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack(
        (cos_latitude, np.sin(longitude), np.cos(longitude)),
        axis=1,
    ).astype(np.float32)


def _edge_features(
    sender_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Receiver-local relative position and length, normalized by max length."""
    send = np.asarray(sender_xyz[edges[:, 0]], dtype=np.float64)
    recv = np.asarray(receiver_xyz[edges[:, 1]], dtype=np.float64)
    relative = send - recv
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    scale = max(float(distance.max()), 1e-12)

    longitude = np.arctan2(recv[:, 1], recv[:, 0])
    cos_longitude = np.cos(longitude)
    sin_longitude = np.sin(longitude)
    sin_latitude = np.clip(recv[:, 2], -1.0, 1.0)
    cos_latitude = np.sqrt(np.maximum(1.0 - sin_latitude**2, 0.0))

    radial = recv
    east = np.stack(
        (-sin_longitude, cos_longitude, np.zeros_like(longitude)),
        axis=1,
    )
    north = np.stack(
        (
            -sin_latitude * cos_longitude,
            -sin_latitude * sin_longitude,
            cos_latitude,
        ),
        axis=1,
    )
    local_relative = np.stack(
        (
            np.einsum("ij,ij->i", relative, radial),
            np.einsum("ij,ij->i", relative, east),
            np.einsum("ij,ij->i", relative, north),
        ),
        axis=1,
    )
    return np.concatenate((local_relative / scale, distance / scale), axis=1).astype(
        np.float32
    )


@lru_cache(maxsize=2)
def _cached_connectivity(
    latitudes: tuple[float, ...],
    longitudes: tuple[float, ...],
    mesh_refinement: int,
    radius_query_fraction_edge_length: float,
    mesh2grid_candidate_faces: int,
) -> tuple[GraphConnectivity, tuple[np.ndarray, ...]]:
    hierarchy = get_hierarchy(mesh_refinement)
    finest_mesh = hierarchy[-1]
    mesh_edges = get_multimesh_edges(hierarchy)
    conn = build_connectivity(
        latitudes=np.asarray(latitudes, dtype=np.float32),
        longitudes=np.asarray(longitudes, dtype=np.float32),
        finest_mesh=finest_mesh,
        mesh_edges=mesh_edges,
        radius_query_fraction_edge_length=radius_query_fraction_edge_length,
        mesh2grid_candidate_faces=mesh2grid_candidate_faces,
    )
    static_features = (
        _position_features(conn.grid_xyz),
        _position_features(conn.mesh_xyz),
        _edge_features(conn.grid_xyz, conn.mesh_xyz, conn.grid2mesh_edges),
        _edge_features(conn.mesh_xyz, conn.mesh_xyz, conn.mesh_edges),
        _edge_features(conn.mesh_xyz, conn.grid_xyz, conn.mesh2grid_edges),
    )
    return conn, static_features


class GraphCastSLASolver(nn.Module):
    """Non-autoregressive 15-day SLA reconstruction on a GraphCast multi-mesh.

    The temporal window is represented as input/output channels and reconstructed
    in one pass. This deliberately retains future context and is not a GraphCast
    forecasting rollout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mesh_refinement: int = 4,
        latent_size: int = 128,
        processor_layers: int = 8,
        hidden_layers: int = 1,
        radius_query_fraction_edge_length: float = 0.6,
        aggregate: str = "sum",
        use_input_mask: bool = True,
        use_latlon_features: bool = True,
        predict_residual: bool = True,
        mesh2grid_candidate_faces: int = 8,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mesh_refinement = mesh_refinement
        self.radius_query_fraction_edge_length = radius_query_fraction_edge_length
        self.mesh2grid_candidate_faces = mesh2grid_candidate_faces
        self.use_input_mask = use_input_mask
        self.use_latlon_features = use_latlon_features
        self.predict_residual = predict_residual

        grid_feature_dim = in_channels
        if use_input_mask:
            grid_feature_dim += in_channels
        if use_latlon_features:
            grid_feature_dim += 3
        mesh_feature_dim = 3 if use_latlon_features else 0
        edge_feature_dim = 4
        hidden_dim = latent_size

        def embedder(feature_dim: int) -> nn.Module:
            return make_mlp(
                feature_dim,
                latent_size,
                hidden_dim,
                hidden_layers,
                dropout=dropout,
                layer_norm=layer_norm,
            )

        self.grid_node_embedder = embedder(grid_feature_dim)
        self.mesh_node_embedder = embedder(mesh_feature_dim)
        self.mesh_edge_embedder = embedder(edge_feature_dim)
        self.grid2mesh_edge_embedder = embedder(edge_feature_dim)
        self.mesh2grid_edge_embedder = embedder(edge_feature_dim)

        self.grid2mesh = BipartiteInteractionNetwork(
            latent_dim=latent_size,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            aggregate=aggregate,
            dropout=dropout,
            layer_norm=layer_norm,
            update_sender_nodes=True,
        )
        self.processor = nn.ModuleList(
            [
                MeshInteractionNetwork(
                    latent_dim=latent_size,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    aggregate=aggregate,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for _ in range(processor_layers)
            ]
        )
        self.mesh2grid = BipartiteInteractionNetwork(
            latent_dim=latent_size,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            aggregate=aggregate,
            dropout=dropout,
            layer_norm=layer_norm,
            update_sender_nodes=False,
        )
        self.decoder = make_mlp(
            latent_size,
            out_channels,
            hidden_dim,
            hidden_layers,
            dropout=dropout,
            layer_norm=False,
        )

    def prior_cost(self, state: torch.Tensor) -> torch.Tensor:
        return state.new_zeros(())

    @staticmethod
    def _coordinate_tuple(
        values: torch.Tensor,
        name: str,
        expected_size: int,
        batch_size: int,
    ) -> tuple[float, ...]:
        coordinates = torch.as_tensor(values).detach()
        if coordinates.ndim == 2:
            if coordinates.shape[0] != batch_size:
                raise ValueError(
                    f"batch.{name} has incompatible shape {tuple(coordinates.shape)}"
                )
            reference = coordinates[0]
            if not torch.equal(coordinates, reference.unsqueeze(0).expand_as(coordinates)):
                raise ValueError(f"all samples in a batch must share the same {name} coordinates")
            coordinates = reference
        if coordinates.ndim != 1 or coordinates.numel() != expected_size:
            raise ValueError(
                f"expected batch.{name} with {expected_size} values, "
                f"got shape {tuple(coordinates.shape)}"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError(f"batch.{name} contains non-finite coordinates")
        return tuple(coordinates.to(device="cpu", dtype=torch.float32).tolist())

    def _connectivity(
        self,
        batch,
        height: int,
        width: int,
        batch_size: int,
    ) -> tuple[GraphConnectivity, tuple[np.ndarray, ...]]:
        if not hasattr(batch, "lat") or not hasattr(batch, "lon"):
            raise ValueError("GraphCastSLASolver requires batch.lat and batch.lon")
        latitudes = self._coordinate_tuple(batch.lat, "lat", height, batch_size)
        longitudes = self._coordinate_tuple(batch.lon, "lon", width, batch_size)
        return _cached_connectivity(
            latitudes,
            longitudes,
            self.mesh_refinement,
            self.radius_query_fraction_edge_length,
            self.mesh2grid_candidate_faces,
        )

    @staticmethod
    def _tensor(
        array: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(array, device=device)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _features(
        self,
        x: torch.Tensor,
        grid_position_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        finite_mask = torch.isfinite(x)
        x0 = torch.nan_to_num(x, nan=0.0)
        features = [x0.flatten(2).transpose(1, 2)]
        if self.use_input_mask:
            features.append(finite_mask.to(dtype=x.dtype).flatten(2).transpose(1, 2))
        if self.use_latlon_features:
            features.append(
                grid_position_features.unsqueeze(0).expand(batch_size, -1, -1)
            )
        return torch.cat(features, dim=-1)

    def forward(self, batch) -> torch.Tensor:
        x = batch.input
        if x.ndim != 4:
            raise ValueError(
                f"expected batch.input with shape (B, T, H, W), got {tuple(x.shape)}"
            )
        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")

        conn, static_features = self._connectivity(
            batch,
            height,
            width,
            batch_size,
        )
        (
            grid_position_features_np,
            mesh_position_features_np,
            grid2mesh_features_np,
            mesh_features_np,
            mesh2grid_features_np,
        ) = static_features
        device, dtype = x.device, x.dtype
        grid_position_features = self._tensor(
            grid_position_features_np, device, dtype
        )
        mesh_position_features = self._tensor(
            mesh_position_features_np, device, dtype
        )

        grid2mesh_edges = self._tensor(conn.grid2mesh_edges, device, torch.long)
        mesh_edges = self._tensor(conn.mesh_edges, device, torch.long)
        mesh2grid_edges = self._tensor(conn.mesh2grid_edges, device, torch.long)

        grid2mesh_features = self._tensor(grid2mesh_features_np, device, dtype)
        mesh_features = self._tensor(mesh_features_np, device, dtype)
        mesh2grid_features = self._tensor(mesh2grid_features_np, device, dtype)

        raw_grid_nodes = self._features(x, grid_position_features)
        if self.use_latlon_features:
            raw_mesh_nodes = mesh_position_features.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            raw_mesh_nodes = x.new_zeros(
                batch_size,
                conn.mesh_xyz.shape[0],
                0,
            )

        latent_grid = self.grid_node_embedder(raw_grid_nodes)
        latent_mesh = self.mesh_node_embedder(raw_mesh_nodes)
        latent_grid2mesh_edges = self.grid2mesh_edge_embedder(
            grid2mesh_features
        ).unsqueeze(0).expand(batch_size, -1, -1)
        latent_mesh_edges = self.mesh_edge_embedder(
            mesh_features
        ).unsqueeze(0).expand(batch_size, -1, -1)
        latent_mesh2grid_edges = self.mesh2grid_edge_embedder(
            mesh2grid_features
        ).unsqueeze(0).expand(batch_size, -1, -1)

        latent_grid, latent_mesh, latent_grid2mesh_edges = self.grid2mesh(
            latent_grid,
            latent_mesh,
            latent_grid2mesh_edges,
            grid2mesh_edges[:, 0],
            grid2mesh_edges[:, 1],
        )
        for block in self.processor:
            latent_mesh, latent_mesh_edges = block(
                latent_mesh,
                latent_mesh_edges,
                mesh_edges[:, 0],
                mesh_edges[:, 1],
            )

        _, latent_grid, latent_mesh2grid_edges = self.mesh2grid(
            latent_mesh,
            latent_grid,
            latent_mesh2grid_edges,
            mesh2grid_edges[:, 0],
            mesh2grid_edges[:, 1],
        )
        decoded = self.decoder(latent_grid)
        decoded = decoded.transpose(1, 2).reshape(
            batch_size,
            self.out_channels,
            height,
            width,
        )
        if self.predict_residual:
            decoded = torch.nan_to_num(x, nan=0.0)[:, : self.out_channels] + decoded
        return decoded

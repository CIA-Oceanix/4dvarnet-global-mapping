import torch
import torch.nn as nn

from contrib.graph_unet.graph_utils import (
    graph_to_grid,
    grid_4_neighbour_edge_index,
    grid_to_batched_graph,
)

try:
    from contrib.graph_unet.source_code_res import GraphUNet
except ModuleNotFoundError:
    from source_code_res import GraphUNet

class GraphUNetDirectSolver(nn.Module):
    """
    Simple baseline: grid -> graph -> PyG GraphUNet (modified version) -> grid.

    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=96,
        depth=3,
        pool_ratios=0.5,
        dropout=0.0,
        act="relu",
        sum_res=True,
        time_embed_dim=None,
    ):
        super().__init__()

        if GraphUNet is None:
            raise ImportError(
                "GraphUNetDirectSolver requires torch-geometric. "
                "Install torch-geometric on the server before using this config."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self._edge_index_cache = {}

        if isinstance(act, str):
            a = act.lower()
            if a in ("silu", "swish"):
                act = nn.SiLU()
            elif a == "relu":
                act = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation function: {act}")

        self.graph_unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=self.out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=act,
            sum_res=sum_res,
            time_embed_dim=time_embed_dim,
        )

    def _get_edge_index(self, height, width, device):
        key = (height, width, device.type, device.index)
        if key not in self._edge_index_cache:
            self._edge_index_cache[key] = grid_4_neighbour_edge_index(
                height=height,
                width=width,
                device=device,
            )
        return self._edge_index_cache[key]

    def forward(self, batch, timesteps=None):
        grid = batch.input.nan_to_num()
        batch_size, channels, height, width = grid.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")

        edge_index = self._get_edge_index(height, width, grid.device)
        nodes, batched_edge_index, graph_batch = grid_to_batched_graph(
            grid=grid,
            edge_index=edge_index,
        )

        if timesteps is None:
            timesteps = getattr(batch, "timesteps", None)
        if timesteps is not None:
            timesteps = torch.as_tensor(timesteps, device=grid.device)
            if timesteps.ndim == 0:
                timesteps = timesteps.expand(batch_size)
            if timesteps.ndim != 1 or timesteps.size(0) != batch_size:
                raise ValueError(
                    "Expected one timestep per grid sample, got shape "
                    f"{tuple(timesteps.shape)} for batch size {batch_size}"
                )

        nodes = self.graph_unet(
            nodes,
            batched_edge_index,
            batch=graph_batch,
            edge_weight=None,
            timesteps=timesteps,
        )

        nodes = self.dropout(nodes)

        return graph_to_grid(
            nodes=nodes,
            batch_size=batch_size,
            channels=self.out_channels,
            height=height,
            width=width,
        )

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _cached_grid_edge_index(height, width):
    num_nodes = height * width
    nodes = torch.arange(num_nodes, dtype=torch.long).reshape(height, width)

    right_src = nodes[:, :-1].reshape(-1)
    right_dst = nodes[:, 1:].reshape(-1)
    down_src = nodes[:-1, :].reshape(-1)
    down_dst = nodes[1:, :].reshape(-1)

    src = torch.cat((right_src, right_dst, down_src, down_dst), dim=0)
    dst = torch.cat((right_dst, right_src, down_dst, down_src), dim=0)

    return torch.stack((src, dst), dim=0)


def grid_4_neighbour_edge_index(height, width, device=None):
    edge_index = _cached_grid_edge_index(int(height), int(width))
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def make_graph_batch(edge_index, num_nodes, batch_size, device=None):
    if device is None:
        device = edge_index.device

    edge_index = edge_index.to(device)
    offsets = torch.arange(batch_size, device=device, dtype=edge_index.dtype) * num_nodes
    batched_edge_index = edge_index.unsqueeze(0) + offsets[:, None, None]
    batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, -1)

    batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
    return batched_edge_index, batch


def grid_to_graph(grid):
    if grid.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(grid.shape)}")

    batch_size, channels, height, width = grid.shape
    nodes = grid.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
    return nodes


def graph_to_grid(nodes, batch_size, channels, height, width):
    expected_nodes = batch_size * height * width
    if nodes.shape != (expected_nodes, channels):
        raise ValueError(
            "Expected graph tensor shape "
            f"{(expected_nodes, channels)}, got {tuple(nodes.shape)}"
        )

    return nodes.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)


def grid_to_batched_graph(grid, edge_index=None):
    if grid.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(grid.shape)}")

    batch_size, _, height, width = grid.shape
    num_nodes = height * width

    if edge_index is None:
        edge_index = grid_4_neighbour_edge_index(height, width, device=grid.device)

    nodes = grid_to_graph(grid)
    batched_edge_index, batch = make_graph_batch(
        edge_index=edge_index,
        num_nodes=num_nodes,
        batch_size=batch_size,
        device=grid.device,
    )

    return nodes, batched_edge_index, batch

"""PyTorch message-passing layers used by the global graph SLA model."""

from __future__ import annotations

import torch
from torch import nn


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float = 0.0,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def aggregate_messages(
    messages: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
    aggregate: str,
) -> torch.Tensor:
    out = messages.new_zeros(messages.shape[0], num_nodes, messages.shape[-1])
    index = receivers.view(1, -1, 1).expand(messages.shape[0], -1, messages.shape[-1])
    out.scatter_add_(1, index, messages)
    if aggregate == "mean":
        counts = messages.new_zeros(num_nodes)
        counts.scatter_add_(0, receivers, torch.ones_like(receivers, dtype=messages.dtype))
        out = out / counts.clamp_min(1.0).view(1, -1, 1)
    elif aggregate != "sum":
        raise ValueError(f"unsupported aggregate: {aggregate}")
    return out


class BipartiteInteractionNetwork(nn.Module):
    """GraphCast interaction network for one directed bipartite subgraph."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        aggregate: str = "sum",
        dropout: float = 0.0,
        layer_norm: bool = True,
        update_sender_nodes: bool = False,
    ):
        super().__init__()
        self.aggregate = aggregate
        self.update_sender_nodes = update_sender_nodes
        self.edge_mlp = make_mlp(
            3 * latent_dim,
            latent_dim,
            hidden_dim,
            hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.receiver_node_mlp = make_mlp(
            2 * latent_dim,
            latent_dim,
            hidden_dim,
            hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.sender_node_mlp = (
            make_mlp(
                latent_dim,
                latent_dim,
                hidden_dim,
                hidden_layers,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            if update_sender_nodes
            else None
        )

    def forward(
        self,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
        edge_latents: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sender_feat = sender_nodes[:, senders, :]
        receiver_feat = receiver_nodes[:, receivers, :]
        edge_update = self.edge_mlp(
            torch.cat((edge_latents, sender_feat, receiver_feat), dim=-1)
        )
        edge_latents = edge_latents + edge_update
        aggregated = aggregate_messages(
            edge_latents,
            receivers,
            receiver_nodes.shape[1],
            self.aggregate,
        )
        receiver_update = self.receiver_node_mlp(
            torch.cat((receiver_nodes, aggregated), dim=-1)
        )
        receiver_nodes = receiver_nodes + receiver_update
        if self.sender_node_mlp is not None:
            sender_nodes = sender_nodes + self.sender_node_mlp(sender_nodes)
        return sender_nodes, receiver_nodes, edge_latents


class MeshInteractionNetwork(nn.Module):
    """One GraphCast processor block with persistent node and edge states."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        aggregate: str = "sum",
        dropout: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.aggregate = aggregate
        self.edge_mlp = make_mlp(
            3 * latent_dim,
            latent_dim,
            hidden_dim,
            hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.node_mlp = make_mlp(
            2 * latent_dim,
            latent_dim,
            hidden_dim,
            hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_latents: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender_feat = nodes[:, senders, :]
        receiver_feat = nodes[:, receivers, :]
        edge_update = self.edge_mlp(
            torch.cat((edge_latents, sender_feat, receiver_feat), dim=-1)
        )
        edge_latents = edge_latents + edge_update
        aggregated = aggregate_messages(
            edge_latents,
            receivers,
            nodes.shape[1],
            self.aggregate,
        )
        node_update = self.node_mlp(torch.cat((nodes, aggregated), dim=-1))
        nodes = nodes + node_update
        return nodes, edge_latents

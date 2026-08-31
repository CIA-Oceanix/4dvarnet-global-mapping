import math
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat


def timestep_embedding(timesteps: Tensor, dim: int,
                       max_period: int = 10000) -> Tensor:
    """Create sinusoidal embeddings for one timestep per graph."""
    half = dim // 2
    if half == 0:
        return timesteps[:, None].float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class GraphResBlock(torch.nn.Module):
    """A timestep-conditioned residual block with two graph convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = activation_resolver(act)

        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.conv1 = GCNConv(in_channels, out_channels, improved=True)
        self.time_proj = torch.nn.Linear(time_embed_dim, out_channels)
        self.norm2 = torch.nn.LayerNorm(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels, improved=True)
        self.norm3 = torch.nn.LayerNorm(out_channels)
        self.conv3 = GCNConv(out_channels, out_channels, improved=True)
        self.skip = (
            torch.nn.Identity()
            if in_channels == out_channels
            else torch.nn.Linear(in_channels, out_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.conv1.reset_parameters()
        self.time_proj.reset_parameters()
        self.norm2.reset_parameters()
        self.conv2.reset_parameters()
        self.norm3.reset_parameters()
        self.conv3.reset_parameters()
        if hasattr(self.skip, 'reset_parameters'):
            self.skip.reset_parameters()

        # Start each block as its skip path, as in the grid U-Net ResBlock.
        for parameter in self.conv3.parameters():
            torch.nn.init.zeros_(parameter)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch: Tensor,
        emb: Optional[Tensor],
    ) -> Tensor:
        residual = self.skip(x)

        h = self.conv1(self.act(self.norm1(x)), edge_index, edge_weight)
        if emb is not None:
            h = h + self.time_proj(emb)[batch]
        h = self.conv2(self.act(self.norm2(h)), edge_index, edge_weight)
        h = self.conv3(self.act(self.norm3(h)), edge_index, edge_weight)
        return residual + h


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
        time_embed_dim (int, optional): Size of the shared timestep embedding.
            Defaults to four times :obj:`hidden_channels`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.sum_res = sum_res
        self.time_embed_dim = time_embed_dim or 4 * hidden_channels

        channels = hidden_channels
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(channels, self.time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.down_blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_blocks.append(
            GraphResBlock(
                in_channels,
                channels,
                self.time_embed_dim,
                act,
            )
        )
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_blocks.append(
                GraphResBlock(
                    channels,
                    channels,
                    self.time_embed_dim,
                    act,
                )
            )

        in_channels = channels if sum_res else 2 * channels

        self.up_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            self.up_blocks.append(
                GraphResBlock(
                    in_channels,
                    channels,
                    self.time_embed_dim,
                    act,
                )
            )
        self.out_norm = torch.nn.LayerNorm(channels)
        self.out_act = activation_resolver(act)
        self.out_conv = GCNConv(channels, out_channels, improved=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for block in self.down_blocks:
            block.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for block in self.up_blocks:
            block.reset_parameters()
        self.out_norm.reset_parameters()
        self.out_conv.reset_parameters()
        for parameter in self.out_conv.parameters():
            torch.nn.init.zeros_(parameter)
        for module in self.time_embed:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: OptTensor = None,
        edge_weight: Tensor = None,
        timesteps: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the U-Net with an optional timestep for each graph in batch."""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        assert edge_weight.dim() == 1
        assert edge_weight.size(0) == edge_index.size(1)

        emb = None
        if timesteps is not None:
            if timesteps.dim() == 0:
                timesteps = timesteps[None]
            if timesteps.dim() != 1:
                raise ValueError('timesteps must be a scalar or a 1-D tensor')
            num_graphs = int(batch.max()) + 1
            if timesteps.size(0) != num_graphs:
                raise ValueError(
                    f'Expected {num_graphs} timesteps, got '
                    f'{timesteps.size(0)}'
                )
            emb = self.time_embed(
                timestep_embedding(timesteps, self.hidden_channels).to(x)
            )

        x = self.down_blocks[0](x, edge_index, edge_weight, batch, emb)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        batches = [batch]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_blocks[i](x, edge_index, edge_weight, batch, emb)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
                batches += [batch]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            batch = batches[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_blocks[i](x, edge_index, edge_weight, batch, emb)

        return self.out_conv(self.out_act(self.out_norm(x)), edge_index,
                             edge_weight)

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')

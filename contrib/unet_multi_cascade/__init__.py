import torch.nn as nn
import torch
torch.use_deterministic_algorithms(True, warn_only=True)

import numpy as np
from torch.nn.functional import relu
import torch.nn.functional as F
#from contrib.unet.parts import StandardBlock, ResBlock, Down, Up, OutConv, Dense, GaussianFourierProjection
from ocean4dvarnet.models import GradSolver
import kornia.filters as kfilts
import math
from collections import namedtuple


TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])






def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.grad_mod_low.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
            {"params": lit_mod.solver.prior_cost_low.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }





class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(max(1, out_ch // 8), out_ch)

        self.act = lambda x: x * torch.sigmoid(x)  # Swish
        self.dense = Dense(embed_dim, out_ch)      # time embedding projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # skip 1x1 conv si dimensions changent
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, embed):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + self.dense(embed))
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + self.dense(embed))

        return h + self.skip(x)


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    ret = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return ret

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class UnetGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, embed_dim, num_levels):
        super().__init__()

        channels = [dim_hidden]
        for i in range(num_levels - 1):
            channels.append(channels[i] * 2)

        # Embedding temporel
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.act = lambda x: x * torch.sigmoid(x)
        self.dropout = nn.Dropout(0.1)
        self.norm = torch.nn.Parameter(torch.tensor([1.]))

        # Encoding
        self.enc_blocks = nn.ModuleList()
        in_ch = dim_in 
        for ch in channels:
            self.enc_blocks.append(ResBlock(in_ch, ch, embed_dim))
            in_ch = ch

        # Bottleneck avec CBAM
        self.bottleneck = ResBlock(channels[-1], channels[-1], embed_dim, dropout=0.1)


        # Decoding
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(1, len(channels))):
            self.dec_blocks.append(
                nn.ModuleDict({
                    "upsample": nn.ConvTranspose2d(channels[i], channels[i-1], 4, stride=2, padding=1),
                    "resblock": ResBlock(channels[i-1]*2, channels[i-1], embed_dim)
                })
            )


        # Projection finale
        self.final_conv = nn.Conv2d(channels[0], dim_in, 3, padding=1)

    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x, t):
        if self._grad_norm is None:
            self._grad_norm = (x ** 2).mean().sqrt()
        x = x / self._grad_norm

        # time embedding 
        embed = self.act(self.embed(t))

        # --- Encoder ---
        hs = []
        h = x
        for block in self.enc_blocks:
            h = block(h, embed)
            hs.append(h)
            h = F.avg_pool2d(h, 2) if block != self.enc_blocks[-1] else h  # downsample sauf dernier

        

        # Bottleneck
        h = self.bottleneck(h, embed)

        # --- Decoder ---
        skip_connections = hs[::-1]
        for skip, dec in zip(skip_connections[1:], self.dec_blocks):  # on garde aussi skip du niveau le + bas
            h = dec["upsample"](h)
            h = dec["resblock"](torch.cat([h, skip], dim=1), embed)


        # Projection finale
        return self.final_conv(h)




class GradSolver_cascade(GradSolver):  
    def __init__(self, prior_cost, prior_cost_low, obs_cost,
                 grad_mod, grad_mod_low, n_step, lr_grad, **kwargs):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, **kwargs)  

        self.grad_mod_high = grad_mod
        self.grad_mod_low = grad_mod_low
        self.prior_cost_high = prior_cost
        self.prior_cost_low = prior_cost_low
        self._grad_norm = None


    def solver_step(self, state, input, step, grad_mod, prior_cost):
        var_cost = prior_cost(state) + self.obs_cost(state, input)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        t = torch.tensor([step], device=grad.device).repeat(grad.shape[0])
        gmod = grad_mod(grad, t)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update 
    
    def forward(self, batch):
        with torch.set_grad_enabled(True):
            y = batch.input
            #Création basse réolution
            # 1. Masquer les NaNs
            mask = ~torch.isnan(y)
            y_masked = torch.nan_to_num(y, nan=0.0)
            size_level = y_masked.shape[2] / 4
            # 2. Pooling pondéré
            y_sum_pool = F.avg_pool2d(y_masked * mask, 4, stride=4) * size_level
            count_pool = F.avg_pool2d(mask.float(), 4, stride=4) * size_level
            y_avg_pooled = y_sum_pool / count_pool
            y_avg_pooled[count_pool == 0] = float('nan')
            state_low = torch.zeros_like(y_avg_pooled).detach().requires_grad_(True)
            residus =  torch.zeros_like(y).detach().requires_grad_(True)
            self.grad_mod_high.reset_state(y)
            self.grad_mod_low.reset_state(y)
            for step in range(self.n_step):
                ## Basse résolution
                state_low = self.solver_step(state_low, y_avg_pooled, step=step,grad_mod = self.grad_mod_low,prior_cost=self.prior_cost_low)
                state_low_out = F.interpolate(state_low, size=y.shape[2:], mode="bilinear", align_corners=False)
                diff = y - state_low_out
                residus = self.solver_step(residus, diff, step=step, grad_mod = self.grad_mod_high,prior_cost=self.prior_cost_high) 
                state = state_low_out + residus
                state_low = F.avg_pool2d(state, 4, stride=4)
                if not self.training:
                    state_low = state_low.detach().requires_grad_(True)
                    residus = residus.detach().requires_grad_(True)
            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)
        return state


class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, input):
        msk = input.isfinite()
        return self.w * F.mse_loss(state[msk], input.nan_to_num()[msk])



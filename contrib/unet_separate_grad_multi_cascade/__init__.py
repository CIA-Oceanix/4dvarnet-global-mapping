import torch.nn as nn
import torch
torch.use_deterministic_algorithms(True, warn_only=True)

import numpy as np
from torch.nn.functional import relu
import torch.nn.functional as F
#from contrib.unet.parts import StandardBlock, ResBlock, Down, Up, OutConv, Dense, GaussianFourierProjection
from contrib.unet import GaussianFourierProjection, Dense
from ocean4dvarnet.models import GradSolver
from contrib.unet import Lit4dVarNetIgnoreNaN
import kornia.filters as kfilts
import math
from collections import namedtuple


TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])






def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod_high.parameters(), "lr": lr},
            {"params": lit_mod.solver.grad_mod_low.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost_high.parameters(), "lr": lr / 2},
            {"params": lit_mod.solver.prior_cost_low.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (B, C, 1, 1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = self.sigmoid(self.conv(x_cat))  # (B, 1, H, W)
        return x * attn

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
    


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


class UnetGradModel_Sep_Grad(nn.Module):
    def __init__(self, dim_in, dim_hidden, embed_dim, num_levels, interleave=True):
        super().__init__()

        self.interleave = interleave
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
        in_ch = dim_in * 2
        for ch in channels:
            self.enc_blocks.append(ResBlock(in_ch, ch, embed_dim))
            in_ch = ch

        # Bottleneck avec CBAM
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], embed_dim, dropout=0.1),
            CBAM(channels[-1]),
        ])

        # Decoding
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(1, len(channels))):
            self.dec_blocks.append(
                nn.ModuleDict({
                    "upsample": nn.ConvTranspose2d(channels[i], channels[i-1], 4, stride=2, padding=1),
                    "resblock": ResBlock(channels[i-1]*2, channels[i-1], embed_dim)
                })
            )

        # CBAM haut niveau
        #self.top_att = CBAM(channels[0]*2)

        # Projection finale
        self.final_conv = nn.Conv2d(channels[0], dim_in, 3, padding=1)

    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x1, x2, x, t):
        B, C, H, W = x1.shape

        # Normalisation
        if self._grad_norm is None:
            self._grad_norm = (x ** 2).mean().sqrt()
        x1 = x1 / self._grad_norm
        x2 = x2 / self._grad_norm

        # Embedding temporel
        embed = self.act(self.embed(t))

        # Fusion entrée
        if self.interleave:
            x1_split, x2_split = torch.chunk(x1, chunks=C, dim=1), torch.chunk(x2, chunks=C, dim=1)
            interleaved = [ch for pair in zip(x1_split, x2_split) for ch in pair]
            h = torch.cat(interleaved, dim=1)
        else:
            h = torch.cat([x1, x2], dim=1)

        # Encoding
        hs = []
        for block in self.enc_blocks:
            h = block(h, embed)
            hs.append(h)
            h = F.avg_pool2d(h, 2) if block != self.enc_blocks[-1] else h  # downsample sauf dernier

        

        # Bottleneck
        for m in self.bottleneck:
            if isinstance(m, ResBlock):
                h = m(h, embed)
            else:
                h = m(h)

        # --- Decoder ---
        skip_connections = hs[::-1]
        for skip, dec in zip(skip_connections[1:], self.dec_blocks):  # on garde aussi skip du niveau le + bas
            h = dec["upsample"](h)
            h = dec["resblock"](torch.cat([h, skip], dim=1), embed)


        # Attention haut niveau
        #h = self.top_att(h)

        # Projection finale
        return self.final_conv(h)



class GradSolver_separate_grad(GradSolver):  
    def __init__(self, prior_cost, prior_cost_low, obs_cost,
                 grad_mod, grad_mod_low, n_step, lr_grad, **kwargs):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, **kwargs)  

        self.grad_mod_high = grad_mod
        self.grad_mod_low = grad_mod_low
        self.prior_cost_high = prior_cost
        self.prior_cost_low = prior_cost_low
        self._grad_norm = None


    def solver_step(self, state, input, step, grad_mod, prior_cost):
        obs = self.obs_cost(state,input)
        gobs = torch.autograd.grad(obs, state, create_graph=True)[0]
        prior = prior_cost(state)
        gprior = torch.autograd.grad(prior, state, create_graph=True)[0]
        grad = torch.autograd.grad(obs+prior, state, create_graph=True)[0]
        t = torch.tensor([step], device=grad.device).repeat(grad.shape[0])
        gmod = grad_mod(gprior, gobs, grad, t)
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








# unet_film_temporalattn_3d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities ----------

def _choose_gn_groups(c: int, prefer: int = 32) -> int:
    """Pick the largest number of groups <= prefer that divides c."""
    for g in [prefer, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier time features for continuous-time conditioning.
    """
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B,1) in [0,1]
        returns: (B, embed_dim)
        """
        t = t.view(-1)  # ensure shape (B,)
        x_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimeMLP(nn.Module):
    """
    t -> embedding vector used for FiLM (scale/shift) in ResBlocks.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

# ---------- Core building blocks ----------

class ResBlock3D_FiLM(nn.Module):
    """
    3D residual block with FiLM time conditioning and dropout.
    - Two 3x3x3 convs, GroupNorm, SiLU.
    - FiLM provides per-channel (scale, shift) after first norm.
    - Optional channel change via 1x1x1 skip.
    - Zero-init last conv for stable residual learning.
    """
    def __init__(self, c_in: int, c_out: int, t_dim: int, dropout: float = 0.10):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = nn.GroupNorm(_choose_gn_groups(c_in), c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)

        self.film = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * c_out))

        self.norm2 = nn.GroupNorm(_choose_gn_groups(c_out), c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv3d(c_in, c_out, kernel_size=1)

        # Zero-init final conv to encourage identity at start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, H, W)
        t_emb: (B, t_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM conditioning
        film = self.film(t_emb)  # (B, 2*C_out)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma[..., None, None, None]
        beta  = beta[...,  None, None, None]
        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        return h + self.skip(x)

class TemporalSelfAttention3D(nn.Module):
    """
    Self-attention along the temporal axis only, with pre-norm and residual.
    - Flattens spatial dims (H*W) into batch for efficiency.
    """
    def __init__(self, channels: int, n_heads: int = 4, attn_dropout: float = 0.20):
        super().__init__()
        self.norm = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns: (B, C, T, H, W)
        """
        b, c, t, h, w = x.shape
        h_in = x
        x = self.norm(x)

        # (B, C, T, H, W) -> (B*H*W, T, C)
        x = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, t, c)
        x_out, _ = self.attn(x, x, x, need_weights=False)
        x_out = self.drop(x_out)
        # back to (B, C, T, H, W)
        x_out = x_out.view(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()
        return h_in + x_out

class Downsample3D(nn.Module):
    """
    Simple strided 3D conv downsample that also sets the target channel count.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):
    """
    3D transposed-conv upsample to target channel count.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(
            c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv(x)

# ---------- The UNet ----------


############### ################################################################################################
# 3D U-Net with:
# Flexible input size, no need to be divisble by 8 because of downsampling with stride=2. 
# This lets you use any T,H,W. Pad to multiples of 8 before the network, then unpad the output.
################################################################################################################
def _pad_to_multiple(x, mult=(8,8,8)):
    # x: (B,C,T,H,W); mult: (mT,mH,mW)
    _,_,T,H,W = x.shape
    mT,mH,mW = mult
    pT = (mT - T % mT) % mT
    pH = (mH - H % mH) % mH
    pW = (mW - W % mW) % mW
    # pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
    x = F.pad(x, (0,pW, 0,pH, 0,pT))
    return x, (pT,pH,pW)

def _unpad(x, pads):
    pT,pH,pW = pads
    if pT or pH or pW:
        T,H,W = x.shape[-3:]
        x = x[..., :T-pT, :H-pH, :W-pW]
    return x
    
class UNet3D_Transformer(nn.Module):
    """
    3D U-Net with:
      - Flexible input size, no need to be divisble by 8 because of downsampling with stride=2. This lets you use any T,H,W. Pad to multiples of          8 before the network, then unpad the output.
      - Residual blocks + FiLM time conditioning
      - Two ResBlocks per stage
      - Temporal self-attention at the bottleneck
      - Depth over width: channels = [48, 96, 192, 384]
    Flow-matching ready: no score-based output normalization.

    Inputs:
      x: (B, 1, T, H, W)
      y: (B, 1, T, H, W)  (conditioning; concatenated with x along channel dim)
      t: (B,) or (B,1)    (continuous "time" / flow parameter in [0,1])

    Output:
      (B, 1, T, H, W)
    """
    def __init__(
        self,
        marginal_prob_std=None,             # kept for compatibility; unused in FM
        in_channels = 2,
        channels=(48, 96, 192, 384),        # depth over width
        embed_dim=256,
        dropout=0.10,                       # set to 0.05–0.20 per your dataset size
        attn_heads=4,
        text_dim=1,                         # kept for signature compatibility
        nAttr=40                            # kept for signature compatibility
    ):
        super().__init__()
        self.dim_3d = True
        self.channels = list(channels)
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.attn_heads = attn_heads
        self.marginal_prob_std = marginal_prob_std  # not used (FM)

        # --- Time embedding ---
        self.time_mlp = TimeMLP(embed_dim=embed_dim)

        c1, c2, c3, c4 = self.channels

        # --- Encoder (Down path): 2 ResBlocks per stage ---
        # Stage 1
        self.enc1_block1 = ResBlock3D_FiLM(c_in=in_channels,  c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.enc1_block2 = ResBlock3D_FiLM(c_in=c1, c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.down1 = Downsample3D(c_in=c1, c_out=c2)

        # Stage 2
        self.enc2_block1 = ResBlock3D_FiLM(c_in=c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.enc2_block2 = ResBlock3D_FiLM(c_in=c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.down2 = Downsample3D(c_in=c2, c_out=c3)

        # Stage 3
        self.enc3_block1 = ResBlock3D_FiLM(c_in=c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.enc3_block2 = ResBlock3D_FiLM(c_in=c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.down3 = Downsample3D(c_in=c3, c_out=c4)

        # --- Bottleneck ---
        self.bot_block1 = ResBlock3D_FiLM(c_in=c4, c_out=c4, t_dim=embed_dim, dropout=dropout)
        self.bot_attn   = TemporalSelfAttention3D(channels=c4, n_heads=attn_heads, attn_dropout=dropout)
        self.bot_block2 = ResBlock3D_FiLM(c_in=c4, c_out=c4, t_dim=embed_dim, dropout=dropout)

        # --- Decoder (Up path): upsample -> concat skip -> 2 ResBlocks ---
        # Up from bottleneck to stage 3
        self.up3 = Upsample3D(c_in=c4, c_out=c3)
        self.dec3_block1 = ResBlock3D_FiLM(c_in=c3 + c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.dec3_block2 = ResBlock3D_FiLM(c_in=c3,       c_out=c3, t_dim=embed_dim, dropout=dropout)

        # Up to stage 2
        self.up2 = Upsample3D(c_in=c3, c_out=c2)
        self.dec2_block1 = ResBlock3D_FiLM(c_in=c2 + c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.dec2_block2 = ResBlock3D_FiLM(c_in=c2,       c_out=c2, t_dim=embed_dim, dropout=dropout)

        # Up to stage 1
        self.up1 = Upsample3D(c_in=c2, c_out=c1)
        self.dec1_block1 = ResBlock3D_FiLM(c_in=c1 + c1, c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.dec1_block2 = ResBlock3D_FiLM(c_in=c1,       c_out=c1, t_dim=embed_dim, dropout=dropout)

        # --- Final head ---
        self.out_norm = nn.GroupNorm(_choose_gn_groups(c1), c1)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(c1, 1, kernel_size=3, padding=1)

        # (Optional) attribute embedding kept for signature parity; not used here
        self.cond_embed = nn.Embedding(nAttr + 1, text_dim, padding_idx=nAttr)

    def predict(self, x: torch.Tensor, timesteps: torch.Tensor = None, extra: torch.Tensor = None) -> torch.Tensor:


        return self.forward(x, timesteps, extra)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:

        # 2D to 3D tensors if needed
        if len(x.shape) == 4:
            dim_x = 2
            x = x.unsqueeze(1)  # (B, 1, T, H, W)

            if y is not None :
                y = y.unsqueeze(1)  # (B, 1, T, H, W)
        else:
            dim_x = 3

        #padding x and y to get divisble for 8
        (x, pads) = _pad_to_multiple(x, (8,8,8))

        if y is not None:
            (y, _   ) = _pad_to_multiple(y, (8,8,8))
            h = torch.cat([x, y], dim=1)  # (B, +1, T, H, W)
        else:
            h = x  
        t_emb = self.time_mlp(t)  # (B, embed_dim)

        # ----- Encoder -----
        h1 = self.enc1_block1(h, t_emb)
        h1 = self.enc1_block2(h1, t_emb)
        d1 = self.down1(h1)

        h2 = self.enc2_block1(d1, t_emb)
        h2 = self.enc2_block2(h2, t_emb)
        d2 = self.down2(h2)

        h3 = self.enc3_block1(d2, t_emb)
        h3 = self.enc3_block2(h3, t_emb)
        d3 = self.down3(h3)

        # ----- Bottleneck -----
        b = self.bot_block1(d3, t_emb)
        b = self.bot_attn(b)
        b = self.bot_block2(b, t_emb)

        # ----- Decoder -----
        u3 = self.up3(b)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.dec3_block1(u3, t_emb)
        u3 = self.dec3_block2(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.dec2_block1(u2, t_emb)
        u2 = self.dec2_block2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.dec1_block1(u1, t_emb)
        u1 = self.dec1_block2(u1, t_emb)

        out = self.out_norm(u1)
        out = self.out_act(out)
        out = self.out_conv(out)

        out = _unpad(out, pads)
        # Flow matching: no division by marginal_prob_std(t)

        if dim_x == 2:
            out = out.squeeze(1)  # (B, T, H, W)
        
        return out
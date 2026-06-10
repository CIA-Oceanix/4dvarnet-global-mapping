import importlib

import torch
import torch.nn.functional as F
from ocean4dvarnet.models import BaseObsCost

_mod = importlib.import_module("contrib.4dvarnet_latent.models")
GradSolver_withStep = _mod.GradSolver_withStep


class BaseObsCostWithUpsampling_(BaseObsCost):
    """ObsCost that upsamples the low-res state to obs resolution before MSE.

    batch.input is always the original 0.25-deg observations (the solver
    handles downsampling internally, so the LitModel never sees 1-deg input).
    """

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        state = F.interpolate(
            state, size=batch.input.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class GradSolver_withStep_Downsampling(GradSolver_withStep):
    """GradSolver_withStep with on-the-fly spatial downsampling.

    Receives the original 0.25-deg batch, downsamples to 1-deg internally
    for all solver iterations, then upsamples output back to 0.25-deg.
    The LightningModule (Lit4dVarNetIgnoreNaN) requires no modification.
    """

    def __init__(self, *args, downsamp=None, unet_stride=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsamp = downsamp
        # UNet downsampling factor (2 ** nb_pool_levels); the 1-deg state must be
        # divisible by it, otherwise the encoder/decoder skip connections mismatch.
        self.unet_stride = unet_stride

    def init_state(self, batch, x_init=None):
        x0 = self.std_init * torch.randn_like(batch.input)
        x0 = F.avg_pool2d(x0, self.downsamp)
        return x0.detach().requires_grad_(True)

    def forward(self, batch, x_init=None, h_state=None, phase='test'):
        out_size = batch.input.shape[-2:]
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)

            # Reflect-pad the 1-deg state so both UNets (prior_cost and grad_mod)
            # receive spatial dims divisible by unet_stride. Without this, patches
            # whose downsampled size is not a multiple of unet_stride (e.g. the
            # 648-lat inference patch -> 162 -> odd 81 after one pooling) break the
            # skip-connection concat inside the UNet.
            h, w = state.shape[-2:]
            ph = (-h) % self.unet_stride
            pw = (-w) % self.unet_stride
            if ph > 0 or pw > 0:
                state = F.pad(state, (0, pw, 0, ph), mode="reflect")
                state = state.detach().requires_grad_(True)
                # CRITICAL: pad the obs grid with NaN by the same amount
                # (x downsamp) so the bilinear upsampling inside the obs cost
                # keeps the state geographically aligned with the observations.
                # Without this the padded state (h+ph rows) is stretched onto
                # the unpadded obs grid (h*downsamp rows), shifting structures
                # by up to ph degrees of latitude. NaNs are masked out in the
                # obs cost so the padded band carries no data term.
                batch = batch._replace(input=F.pad(
                    batch.input,
                    (0, pw * self.downsamp, 0, ph * self.downsamp),
                    value=float("nan"),
                ))

            self.init_h_state(batch, h_state=h_state)
            self.grad_mod.reset_state(batch.input)
            for step in range(self.n_step):
                alpha_step = 1. / self.n_step
                state = self.solver_step(state, batch, step=step / self.n_step, alpha_step=alpha_step)
                if (not self.training) and ('grad' in self.input_grad_update):
                    state = state.detach().requires_grad_(True)

            # Drop the padding before upsampling so the output matches the input grid.
            if ph > 0 or pw > 0:
                state = state[..., :h, :w]

        # size= (not scale_factor) guarantees the output lands exactly on the
        # original 0.25-deg grid regardless of avg_pool2d rounding.
        return F.interpolate(state, size=out_size, mode="bilinear", align_corners=False)

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

    def __init__(self, *args, downsamp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsamp = downsamp

    def init_state(self, batch, x_init=None):
        x0 = self.std_init * torch.randn_like(batch.input)
        x0 = F.avg_pool2d(x0, self.downsamp)
        return x0.detach().requires_grad_(True)

    def forward(self, batch, x_init=None, h_state=None, phase='test'):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.init_h_state(batch, h_state=h_state)
            self.grad_mod.reset_state(batch.input)
            for step in range(self.n_step):
                alpha_step = 1. / self.n_step
                state = self.solver_step(state, batch, step=step / self.n_step, alpha_step=alpha_step)
                if (not self.training) and ('grad' in self.input_grad_update):
                    state = state.detach().requires_grad_(True)

        return F.interpolate(state, scale_factor=float(self.downsamp), mode="bilinear", align_corners=False)

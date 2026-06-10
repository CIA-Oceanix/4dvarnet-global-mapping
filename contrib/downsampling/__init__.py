import importlib
from collections import namedtuple

import kornia.filters as kfilts
import torch
import torch.nn.functional as F
from ocean4dvarnet.models import BaseObsCost

# contrib.4dvarnet_latent starts with a digit, so standard import syntax fails.
_mod = importlib.import_module("contrib.4dvarnet_latent.models")
Lit4dVarNetIgnoreNaN = _mod.Lit4dVarNetIgnoreNaN

# Carries the original 0.25-deg input alongside the downsampled one so that
# BaseObsCostWithUpsampling can compare the state to native-res altimeter tracks.
TrainingItemWithOrig = namedtuple("TrainingItemWithOrig", ["input", "tgt", "orig_input"])


class BaseObsCostWithUpsampling(BaseObsCost):
    """ObsCost that upsamples the low-res state to obs resolution before MSE.

    When the solver operates at 1 deg but observations are at 0.25 deg, we apply
    bilinear upsampling U(state) before computing the observation misfit.
    Uses batch.orig_input (0.25 deg) when available, falls back to batch.input.
    """

    def forward(self, state, batch):
        obs = getattr(batch, "orig_input", None)
        if obs is None:
            obs = batch.input
        if state.shape != obs.shape:
            state = F.interpolate(
                state, size=obs.shape[-2:], mode="bilinear", align_corners=False
            )
        msk = obs.isfinite()
        return self.w * F.mse_loss(state[msk], obs.nan_to_num()[msk])


class Lit4dVarNetIgnoreNaNDownsampling(Lit4dVarNetIgnoreNaN):
    """Lit4dVarNetIgnoreNaN with on-the-fly spatial downsampling.

    Pipeline:
      1. D(x0) via avg_pool2d with coverage correction  -> state at 1 deg
      2. Solver iterations at 1 deg
      3. Observation cost: U(state) via bilinear interpolate vs 0.25-deg obs

    Set downsamp=4 in config (K = 1 deg / 0.25 deg).
    """

    def __init__(self, *args, downsamp=None, unet_stride=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsamp = downsamp
        self.unet_stride = unet_stride

    def _ds_rw(self, phase):
        rw = self.get_rec_weight(phase)
        if self.downsamp is None:
            return rw
        return F.avg_pool2d(rw, self.downsamp)

    def forward(self, batch):
        if self.downsamp is not None:
            orig_input = batch.input
            orig_H, orig_W = orig_input.shape[-2:]

            cov = F.avg_pool2d(batch.input.isfinite().float(), self.downsamp)
            inp = (
                F.avg_pool2d(batch.input.nan_to_num(), self.downsamp)
                / cov.clamp(min=1e-5)
            ).masked_fill(cov == 0, float("nan"))

            # Pad so that spatial dims are divisible by unet_stride (2^n_pool_levels).
            H, W = inp.shape[-2:]
            ph = (-H) % self.unet_stride
            pw = (-W) % self.unet_stride
            if ph > 0 or pw > 0:
                inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
                # CRITICAL: pad the 0.25-deg obs grid with NaN by the same
                # amount (x downsamp) so that BaseObsCostWithUpsampling keeps
                # the padded state geographically aligned with the obs.
                # Without this the padded state (H+ph rows) is stretched onto
                # the unpadded obs grid (H*downsamp rows), shifting structures
                # by up to ph degrees of latitude. NaNs are masked out in the
                # obs cost so the padded band carries no data term.
                orig_input = F.pad(
                    orig_input,
                    (0, pw * self.downsamp, 0, ph * self.downsamp),
                    value=float("nan"),
                )

            batch = TrainingItemWithOrig(input=inp, tgt=getattr(batch, 'tgt', None), orig_input=orig_input)
            out = self.solver(batch)

            # Remove padding, then upsample back to 0.25-deg so that inferring.py
            # receives output at the same resolution as the input.
            if ph > 0 or pw > 0:
                out = out[..., :H, :W]
            out = F.interpolate(out, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
            return out

        return self.solver(batch)

    def base_step(self, batch, phase):
        # out is back at 0.25-deg after forward's upsample.
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))
        return loss, out

    def loss_mse(self, batch, out, phase):
        # out is at 0.25-deg; compare directly with the 0.25-deg target.
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )
        return loss, grad_loss

    def loss_prior(self, batch, out, phase):
        # Prior cost was trained at 1-deg: re-downsample before evaluating.
        out_ds = F.avg_pool2d(out, self.downsamp) if self.downsamp else out
        loss_prior_out = self.solver.prior_cost(out_ds)
        tgt_ds = F.avg_pool2d(batch.tgt.nan_to_num(), self.downsamp) if self.downsamp else batch.tgt.nan_to_num()
        loss_prior_tgt = self.solver.prior_cost(tgt_ds)
        return loss_prior_out, loss_prior_tgt

    def test_step(self, batch, batch_idx):
        # out is at 0.25-deg; store everything at the same resolution.
        out = self(batch=batch)
        m, s = self.norm_stats["test"]
        self.test_data.append(
            torch.stack(
                [
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            )
        )

    def on_test_epoch_end(self):
        # out is at 0.25-deg so rec_weight needs no adjustment.
        super().on_test_epoch_end()



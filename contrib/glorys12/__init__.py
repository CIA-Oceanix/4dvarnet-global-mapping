"""
Learning GLORYS12 data
"""

import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet


# Exceptions
# ----------


class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


# Data
# ----


class DistinctNormDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_mask = None

        if isinstance(self.input_da, (tuple, list)):
            self.input_da, self.input_mask = self.input_da[0], self.input_da[1]

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(item):
            return (item - m) / s

        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItem._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
            ],
        )

    def setup(self, stage="test"):
        self.train_ds = LazyXrDataset(
            self.input_da.sel(self.domains["train"]),
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mask=self.input_mask,
        )
        self.val_ds = LazyXrDataset(
            self.input_da.sel(self.domains["val"]),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            mask=self.input_mask,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )


class LazyXrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise_type=None,
        noise=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in ds.dims)
        _shape = (2,) + tuple(ds[k].shape[0] for k in ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        # ds_dims = dict(zip(ds.dims, ds.shape))
        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }
        self._rng = np.random.default_rng()
        self.noise = noise

        if noise_type is not None:
            self.noise_type = noise_type
        else:
            self.noise_type = 'uniform-constant'

        # self.mask = kwargs.get("mask")
        self.mask = None

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(), np.unravel_index(item, tuple(self.ds_size.values()))
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        if self.mask is not None:
            start, stop = sl["time"].start % 365, sl["time"].stop % 365

            if start > stop:
                start -= stop

                stop = None

            sl_mask = sl.copy()

            sl_mask["time"] = slice(start, stop)

            da = self.ds.isel(**sl)

            item = (
                da.to_dataset(name="tgt")
                .assign(input=da.where(self.mask.isel(**sl_mask).values))
                .to_array()
                .sortby("variable")
            )

        else:
            item = (
                self.ds.isel(**sl)
                # .to_array()
                # .sortby('variable')
            )

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        if self.noise is not None:

            if self.noise_type ==  'uniform-constant' :
                noise =  self._rng.uniform(-self.noise, self.noise, item[0].shape).astype(np.float32)

                item[0] = item[0] + noise
            elif self.noise_type ==  'gaussian+uniform' :
                scale = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                noise =  self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + scale * noise
            elif self.noise_type ==  'spatial-perturb' :
                # not yet coded
                item[0] = item[0]

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item



# Model
# -----


class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        _val_rec_weight = kwargs.pop(
            "val_rec_weight",
            kwargs["rec_weight"],
        )
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "val_rec_weight",
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == "val":
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    def on_train_epoch_end(self):
        self.log(
            "n_rejected_batches",
            self._n_rejected_batches,
            on_step=False,
            on_epoch=True,
        )

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            if phase == "val":
                # Log the loss in Gulfstream
                loss_gf = self.weighted_mse(
                    out[:, :, 445:485, 420:460].detach().cpu().data
                    - batch.tgt[:, :, 445:485, 420:460].detach().cpu().data,
                    np.ones_like(out[:, :, 445:485, 420:460].detach().cpu().data),
                )
                self.log(
                    f"{phase}_loss_gulfstream",
                    loss_gf,
                    on_step=False,
                    on_epoch=True,
                )

        return loss, out


# Utils
# -----


def load_glorys12_data(tgt_path, inp_path, tgt_var="zos", inp_var="input"):
    isel = None  # dict(time=slice(-465, -265))

    _start = time.time()

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
    )
    inp = xr.open_dataset(inp_path)[inp_var].isel(isel)

    ds = (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .to_array()
        .sortby("variable")
    )

    print(f">>> Durée de chargement : {time.time() - _start:.4f} s")
    return ds


def load_glorys12_data_on_fly_inp(
    tgt_path,
    inp_path,
    tgt_var="zos",
    inp_var="input",
):
    isel = None  # dict(time=slice(-365 * 2, None))

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
        .rename(latitude="lat", longitude="lon")
    )

    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .isel(isel)
        .rename(latitude="lat", longitude="lon")
    )

    return tgt, inp


def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    print(f"Durée d'apprentissage : {time.time() - start:.3} s")

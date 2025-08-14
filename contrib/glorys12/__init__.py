"""
Learning GLORYS12 data
"""

import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr
import pytorch_lightning as pl

from collections import namedtuple
from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet


# Exceptions
# ----------

TrainingItemwithLonLat = namedtuple('TrainingItemwithLonLat', ['input', 'tgt','lon','lat'])

class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


# Data
# ----

def generate_correlated_fields_np(N, M, L, T_corr, sigma, num_fields=10):
    """
    Generate a series of 2D fields with both spatial and temporal correlations.

    Parameters:
        N (int): Grid size (assumed to be square, NxN).
        L (float): Spatial correlation length.
        T_corr (float): Temporal correlation length.
        sigma (float): Standard deviation of the field.
        num_fields (int): Number of fields to generate (default is 10).
        device (str): Device to run the computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Generated fields of shape (num_fields, N, N).
    """
    # Define the time points for the fields
    time_points = np.linspace(0, num_fields - 1, num_fields)

    # Compute the temporal covariance matrix
    C_temporal = np.exp(-abs(time_points[:, None] - time_points[None, :]) / T_corr)

    # Perform Cholesky decomposition to get the temporal correlation factors
    L_chol = np.linalg.cholesky(C_temporal)

    # Generate independent Gaussian white noise for each time point
    white_noises = np.random.randn(num_fields, N, M )

    # Combine white noise using the Cholesky factors to induce temporal correlation
    temporal_correlated_noises = np.matmul(L_chol, white_noises.reshape(num_fields, -1)).reshape(num_fields, N, M)

    # Generate 2D grid of wavenumbers for spatial correlation
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(M) * M
    k = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    cutoff_mask = (k < 20)  # High-frequency cutoff
    # apply - if we apply the same approach to vorticity, and then obtain 
    # stream function, 
    # Spatial covariance (Power spectrum) for Gaussian covariance
    P_k = np.exp(-0.5 * (k * L)**3)
    P_k[0, 0] = 0.0
    P_k = P_k / np.sum(P_k)

    # Generate fields using Fourier transform
    fields = []
    for i in range(num_fields):
        noise_ft = np.fft.fft2(temporal_correlated_noises[i])

        field_ft = noise_ft * sigma**2 * np.sqrt(P_k) * cutoff_mask
        field = np.fft.ifft2(field_ft).real
        fields.append(field)
    return np.stack(fields)

def warp_field_np(field, dx, dy):
    """
    Warp a 2D field based on displacement fields dx and dy.
    field (torch.Tensor): Input field of shape (batch_size, channels, height, width)
    dx (torch.Tensor): X-displacement field of shape (batch_size, height, width)
    dy (torch.Tensor): Y-displacement field of shape (batch_size, height, width)
    """
    height, width = field.shape
    
    # Create base grid
    xref = np.arange(width) 
    yref = np.arange(height)

    yg, xg  = np.meshgrid(yref, xref, indexing='ij')
    #base_grid = np.stack((x, y), dim=-1).float()
 
    # Add batch dimension and move to the same device as input field
    #base_grid = base_grid.unsqueeze(0).repeat(batch_size,1,1,1).to(field.device)

    # Apply displacements
    x = ( xg + dx )
    y = ( yg + dy )

    x [ x < 0. ] = 0.
    y [ y < 0. ] = 0.

    x [ x > width-1 ]  = width - 1.
    y [ y > height-1 ] = height - 1.


     # Normalize grid to [-1, 1] range
    #sample_grid[..., 0] = 2 * sample_grid[..., 0] / (width) - 1
    #sample_grid[..., 1] = 2 * sample_grid[..., 1] / (height) - 1

    # Perform sampling
    #warped_field = F.grid_sample(field, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    interp = RegularGridInterpolator((yref, xref), field , method="cubic")
    warped_field = interp( (y , x ) )
    #warped_field = warped_field.reshape( x.shape )
    return warped_field

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

class DistinctNormDataModuleWithLonLat(DistinctNormDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage="test"):
        self.train_ds = LazyXrDatasetWithLonLat(
            self.input_da.sel(self.domains["train"]),
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mask=self.input_mask,
        )

        self.val_ds = LazyXrDatasetWithLonLat(
            self.input_da.sel(self.domains["val"]),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            mask=self.input_mask,
        )

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(item):
            return (item - m) / s

        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItemwithLonLat._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
            ],
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
        noise_spatial_perturb=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in self.ds.dims)
        _shape = (2,) + tuple(self.ds[k].shape[0] for k in self.ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        # ds_dims = dict(zip(self.ds.dims, self.ds.shape))
        

        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }
        self._rng = np.random.default_rng()
        self.noise = noise
        self.noise_spatial_perturb = noise_spatial_perturb

        if noise_type is not None:
            self.noise_type = noise_type
        else:
            self.noise_type = 'uniform-constant'

        self.mask = kwargs.get("mask")

        print(self.noise_type,flush=True)

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
            item = self.ds.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        #if self.noise:
        #    noise = np.tile(
        #        self._rng.uniform(-self.noise, self.noise, item[0].shape), (2, 1, 1, 1)
        #    ).astype(np.float32)
        #    item = item + noise

        if self.noise is not None:

            if self.noise_type ==  'uniform-constant' :
                #noise =  self._rng.uniform(-self.noise, self.noise, item[0].shape).astype(np.float32)
                #item[0] = item[0] + noise
                noise =  self._rng.uniform(-self.noise, self.noise, item.shape).astype(np.float32)

                item = item + noise
            elif self.noise_type ==  'gaussian+uniform' :
                scale = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                noise = self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + scale * noise
            elif self.noise_type ==  'spatial-perturb' :
                # patch dimensions
                N = item[0].shape[1]
                M = item[0].shape[2]
                T = item[0].shape[0]

                # parameters of the Gaussian Process
                L = 5.0  # Spatial correlation length
                T_corr = 20.0  # Temporal correlation length
                sigma  = self._rng.uniform(0. , self.noise_spatial_perturb, 1).astype(np.float32)

                # generate space-time random perturbations
                w  = np.matmul( np.hanning( N ).reshape(N,1) , np.hanning( M ).reshape(1,M) )
                dx = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)
                dy = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)

                # compute warped fields from reference field and associated error map
                warped_field = np.zeros_like(field)

                # Perform warping
                for ii in range(T):
                    warped_field[ii,:,:] = warp_field_np(item[1][ii,:,:], dx[ii,:,:], dy[ii,:,:])
                
                # residual error
                error = item[1][ii,:,:] - warped_field

                # apply mask
                noise = np.where( np.isnan(item[0]) , np.nan, error )
                print("..... std of the simulated noise : %.3f"%np.nanstd(noise) )
                
                # adding a white noise
                scale  = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                wnoise = scale * self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + noise + wnoise
        
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item


class LazyXrDatasetWithLonLat(LazyXrDataset):
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise_type=None,
        noise=None,
        noise_spatial_perturb=None,
        *args,
        **kwargs,
    ):
        super().__init__(ds, patch_dims, domain_limits, strides, 
                         postpro_fn, noise_type, noise, noise_spatial_perturb, *args,**kwargs)

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

            item_ = (
                da.to_dataset(name="tgt")
                .assign(input=da.where(self.mask.isel(**sl_mask).values))
                .to_array()
                .sortby("variable")
            )
        else:
            item_ = self.ds.isel(**sl)

        if self.return_coords:
            return item_.coords.to_dataset()[list(self.patch_dims)]

        item = item_.data.astype(np.float32)

        if self.noise is not None:

            if self.noise_type ==  'uniform-constant' :
                #noise =  self._rng.uniform(-self.noise, self.noise, item[0].shape).astype(np.float32)
                #item[0] = item[0] + noise
                noise =  self._rng.uniform(-self.noise, self.noise, item.shape).astype(np.float32)

                item = item + noise
            elif self.noise_type ==  'gaussian+uniform' :
                scale = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                noise = self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + scale * noise
            elif self.noise_type ==  'spatial-perturb' :
                # patch dimensions
                N = item[0].shape[1]
                M = item[0].shape[2]
                T = item[0].shape[0]

                # parameters of the Gaussian Process
                L = 5.0  # Spatial correlation length
                T_corr = 20.0  # Temporal correlation length
                sigma  = self._rng.uniform(0. , self.noise_spatial_perturb, 1).astype(np.float32)

                # generate space-time random perturbations
                w  = np.matmul( np.hanning( N ).reshape(N,1) , np.hanning( M ).reshape(1,M) )
                dx = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)
                dy = w * generate_correlated_fields_np(N, M, L, T_corr, sigma,T)

                # compute warped fields from reference field and associated error map
                warped_field = np.zeros_like(field)

                # Perform warping
                for ii in range(T):
                    warped_field[ii,:,:] = warp_field_np(item[1][ii,:,:], dx[ii,:,:], dy[ii,:,:])
                
                # residual error
                error = item[1][ii,:,:] - warped_field

                # apply mask
                noise = np.where( np.isnan(item[0]) , np.nan, error )
                print("..... std of the simulated noise : %.3f"%np.nanstd(noise) )
                
                # adding a white noise
                scale  = self._rng.uniform(0. , self.noise, 1).astype(np.float32)
                wnoise = scale * self._rng.normal(0., 1. , item[0].shape).astype(np.float32)

                item[0] = item[0] + noise + wnoise

        item = TrainingItemwithLonLat(item[0],
                                      item[1],
                                      item_.coords['lon'].data.astype(np.float32),
                                      item_.coords['lat'].data.astype(np.float32))


        if self.postpro_fn is not None:
            item = self.postpro_fn(item)
            return item
        
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

    
    print('..... Start loading dataset',flush=True)
    
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

    print(f">>> Durée de chargement : {time.time() - _start:.4f} s",flush=True)
    #print(ds,flush=True)

    #print(ds["tgt"].data,flush=True)

    return ds

def load_glorys12_data_on_fly_inp(
    tgt_path,
    inp_path,
    tgt_var="zos",
    inp_var="input",
):
 
    print('..... Start lazy loading',flush=True)

    isel = None  # dict(time=slice(-365 * 2, None))

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    print('..... GT dataset')

    if list(tgt.coords)[1] == 'latitude' :
        tgt = tgt.rename(latitude="lat", longitude="lon")

    print(tgt,flush=True)

    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    if list(inp.coords)[1] == 'latitude' :
        inp = inp.rename(latitude="lat", longitude="lon")

    print('..... Obs dataset')
    print('..... NB: only the mask of the obs data might be used by the datamodule')
    print(inp,flush=True)

    return tgt, inp


def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    print(f"Durée d'apprentissage : {time.time() - start:.3} s")

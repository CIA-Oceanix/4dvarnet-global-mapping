import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import xarray as xr
import ocean4dvarnet.data
import xarray as xr
import os

def get_constant_crop(patch_dims, crop, dim_order=("time", "lat", "lon")):
    """
    Returns a 0/1 crop mask with shape [time, lat, lon] (if lon exists)
    or [time, lat] (if lon is not in patch_dims).

    patch_dims: dict like {time:200, lat:40, lon:1}
    crop: dict like {time:10, lat:5, lon:0}
    """
    # Only keep dims that exist in patch_dims
    dim_order = [d for d in dim_order if d in patch_dims]

    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")

    mask = tuple(
        slice(crop.get(d, 0), -crop.get(d, 0)) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )

    patch_weight[mask] = 1.0
    return patch_weight



def get_triang_time_wei(patch_dims, offset=0, crop=None, dim_order=("time", "lat", "lon")):
    """
    Triangular weight along time, with optional cropping.
    Output is made compatible with your model: [1, time, lat].

    If lon exists, it is squeezed out.
    """
    crop = crop or {}

    pw = get_constant_crop(patch_dims, crop=crop, dim_order=dim_order)
    # pw shape: [time, lat, lon] or [time, lat]

    # Build triangular time ramp (shape [time, 1, 1] or [time, 1])
    T = patch_dims["time"]

    if pw.ndim == 3:
        # [time, lat, lon]
        tri = np.fromfunction(
            lambda t, y, x: (1 - np.abs(offset + 2 * t - T) / T),
            pw.shape,
            dtype=float
        ).astype(np.float32)

        w = tri * pw
        w = np.squeeze(w, axis=-1)   # drop lon -> [time, lat]

    # elif pw.ndim == 2:
    #     # [time, lat]
    #     tri = np.fromfunction(
    #         lambda t, y: (1 - np.abs(offset + 2 * t - T) / T),
    #         pw.shape,
    #         dtype=float
    #     ).astype(np.float32)

    #     w = tri * pw

    else:
        raise ValueError(f"Unexpected pw.ndim={pw.ndim}, pw.shape={pw.shape}")

    # Add channel dim -> [1, time, lat]
    return w[None, ...]


def get_constant_time_wei(patch_dims, offset=0, **crop_kw):
    """
    Returns a constant weighting mask (all ones in the non-cropped region, zeros elsewhere)
    with the same shape as the patch (e.g., [time, lat, lon]).
    """
    patch_dims['time'] = 1
    patch_dims['lat'] = patch_dims['time']
    patch_dims['lon'] = patch_dims['lat']
    return get_constant_crop(patch_dims, crop_kw)


def load_l96_data(path, obs_from_tgt=False):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds['obs'],
            tgt=lambda ds: ds['truth']
        )
    )
    
    return (
        ds[[*ocean4dvarnet.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_l96_data_identity(path, obs_from_tgt=False):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds['truth'],
            tgt=lambda ds: ds['truth']
        )
    )
    
    return (
        ds[[*ocean4dvarnet.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )


def load_l96_data_multi(paths):
    """Load multiple trajectory netCDFs, return list of DataArrays"""
    return [load_l96_data(p) for p in paths]



def rmse_based_scores(ds):

    da_rec = ds["out"]
    da_ref = ds["tgt"]
    da_input = ds["inp"]

    # Nombre de timestamps
    n_time = da_rec.sizes["time"]
    k = int(0.05 * n_time)

    # Trim des 5% début et fin
    if k > 0:
        da_rec = da_rec.isel(time=slice(k, -k))
        da_ref = da_ref.isel(time=slice(k, -k))
        da_input = da_input.isel(time=slice(k, -k))

    # RMSE globale
    rmse = np.sqrt(((da_rec - da_ref) ** 2).mean())

    # RMSE normalisée par timestep
    rmse_t = (
        np.sqrt(((da_rec - da_ref) ** 2).mean(dim=("lon", "lat")))
        / np.sqrt((da_ref ** 2).mean(dim=("lon", "lat")))
    )

    std = rmse_t.std()

    # masque pixels manquants
    mask_missing = da_input.isnull()

    # RMSE uniquement sur pixels manquants
    rmse_missing = np.sqrt(((da_rec - da_ref) ** 2).where(mask_missing).mean())

    return (
        np.round(rmse.values, 5).item(),
        np.round(std.values, 5).item(),
        np.round(rmse_missing.values, 5).item(),
    )






def rmse_ensemble_based_scores(ds):

    # 👉 récupérer toutes les variables "out*"
    out_vars = [v for v in ds.data_vars if v.startswith("out")]

    if len(out_vars) == 0:
        raise ValueError("Aucune variable 'out' trouvée dans le dataset")

    # 👉 moyenne des out
    da_rec = xr.concat([ds[v] for v in out_vars], dim="member").mean(dim="member")

    da_ref = ds["tgt"]
    da_input = ds["inp"]

    # Nombre de timestamps
    n_time = da_rec.sizes["time"]
    k = int(0.05 * n_time)

    # Trim des 5% début et fin
    if k > 0:
        da_rec = da_rec.isel(time=slice(k, -k))
        da_ref = da_ref.isel(time=slice(k, -k))
        da_input = da_input.isel(time=slice(k, -k))

    # RMSE globale
    rmse = np.sqrt(((da_rec - da_ref) ** 2).mean())

    # RMSE normalisée par timestep
    rmse_t = (
        np.sqrt(((da_rec - da_ref) ** 2).mean(dim=("lon", "lat")))
        / np.sqrt((da_ref ** 2).mean(dim=("lon", "lat")))
    )

    std = rmse_t.std()

    # masque pixels manquants
    mask_missing = da_input.isnull()

    # RMSE uniquement sur pixels manquants
    rmse_missing = np.sqrt(((da_rec - da_ref) ** 2).where(mask_missing).mean())

    return (
        np.round(rmse.values, 5).item(),
        np.round(std.values, 5).item(),
        np.round(rmse_missing.values, 5).item(),
    )





def crps_based_scores(ds):

    out_vars = sorted([v for v in ds.data_vars if v.startswith("out")])
    da_ens = xr.concat([ds[v] for v in out_vars], dim="member")

    da_ref = ds["tgt"]
    da_input = ds["inp"]

    # trim
    n_time = da_ref.sizes["time"]
    k = int(0.05 * n_time)

    if k > 0:
        da_ens = da_ens.isel(time=slice(k, -k))
        da_ref = da_ref.isel(time=slice(k, -k))
        da_input = da_input.isel(time=slice(k, -k))

    # =========================
    # CRPS STABLE VERSION
    # =========================

    # term 1: ensemble vs truth
    term1 = np.abs(da_ens - da_ref).mean("member")

    # term 2: ensemble spread (NO expand_dims)
    ens_mean = da_ens.mean("member")
    term2 = np.abs(da_ens - ens_mean).mean("member")

    crps = term1 - 0.5 * term2

    crps_mean = crps.mean()

    mask_missing = da_input.isnull()
    crps_missing = crps.where(mask_missing).mean()

    return (
        np.round(crps_mean.values, 5).item(),
        np.round(crps_missing.values, 5).item(),
    )
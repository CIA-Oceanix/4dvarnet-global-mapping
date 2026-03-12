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

    elif pw.ndim == 2:
        # [time, lat]
        tri = np.fromfunction(
            lambda t, y: (1 - np.abs(offset + 2 * t - T) / T),
            pw.shape,
            dtype=float
        ).astype(np.float32)

        w = tri * pw

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
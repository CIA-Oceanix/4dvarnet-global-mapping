import numpy as np
import xarray as xr
import random as rd
from ocean4dvarnet.data import BaseDataModule, TrainingItem
import functools as ft
import torch
import copy


# Exceptions
# ----------
class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


# Data
# ----
class OSEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_mask = None

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
        self.train_ds = LazyXrDatasetOSE(
            self.input_da,
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mask=self.input_mask,
        )
        self.val_ds = LazyXrDatasetOSE(
            self.input_da,
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


class LazyXrDatasetOSE(torch.utils.data.Dataset):
    def __init__(
        self,
        list_ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise=None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the XrDataset for OSE data.

        Args:
            list_ds (xarray.DataArray): Input data, with patch dims at the end in the dim orders
            sla_altimeters_bool_path (xarray.Dataset):
            patch_dims (dict):  da dimension and sizes of patches to extract.
            domain_limits (dict, optional): da dimension slices of domain, to Limits for selecting
                                            a subset of the domain. for patch extractions
            strides (dict, optional): dims to strides size for patch extraction.(default to one)
            postpro_fn (callable, optional): A function for post-processing extracted patches.
        """

        super().__init__()
        # TODO: variables as parameters
        self.bool_dataset_path = "/Odyssey/public/altimetry_traces/2010_2019/ose_2010_2019_daily_bool_1deg.nc"
        self.altimeters_name_list = ["al", "alg", "c2", "enn", "h2a", "h2ag", "j1g", "j1n", "j2", "j2g", "j2n", "j3", "s3a", "s3b"]
        self.return_coords = False
        self.postpro_fn = postpro_fn

        ##
        self.list_ds = [ds.sel(**(domain_limits or {})) for ds in list_ds]
        ##

        self.patch_dims = patch_dims
        self.strides = strides or {}

        ##
        ds_bools = xr.open_dataset(self.bool_dataset_path).rename(latitude="lat", longitude="lon")
        _dims = ("variable",) + tuple(k for k in list_ds[0].dims)
        # HACK: to be improve
        _shape = (2,) + tuple([ds_bools["time"].shape[0], list_ds[0]["lat"].shape[0], list_ds[0]["lon"].shape[0]])
        ds_dims = dict(zip(_dims, _shape))
        ##

        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }
        self._rng = np.random.default_rng()
        self.noise = noise
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
        sl_bool = {}
        _zip = zip(
            self.ds_size.keys(), np.unravel_index(item, tuple(self.ds_size.values()))
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )
            if dim == "time":
                sl_bool[dim] = slice(
                    self.strides.get(dim, 1) * idx,
                    self.strides.get(dim, 1) * idx + self.patch_dims[dim],
                )
            else:
                sl_bool[dim] = slice(
                    (self.strides.get(dim, 1) * idx) // 4, # Resolution en dur
                    (self.strides.get(dim, 1) * idx + self.patch_dims[dim]) // 4, # Resolution en dur
                )

        # Select altimeters depending on availability
        if self.bool_dataset_path is not None:
            ds_bools = xr.open_dataset(self.bool_dataset_path).rename(latitude="lat", longitude="lon")
            ds_bools_patch = ds_bools.isel(**sl_bool)
            bools = (
                ds_bools_patch
                    .any(dim=['lat', 'lon'])
            )
            available_altimeters = [k for k in bools if all(bools[k])]
            tgt_altimeter = rd.choice(available_altimeters)
            index_tgt_altimeter = self.altimeters_name_list.index(tgt_altimeter)
        else:
            raise Exception()


        if self.mask is not None:
            raise Exception("wip")
        else:
            list_input = copy.deepcopy(self.list_ds)
            tgt_altimeter_ds = list_input.pop(index_tgt_altimeter)
            list_input = [
                ds.sel(
                    time=slice(ds_bools_patch["time"].values[0], ds_bools_patch["time"].values[-1]),
                    lat=sl["lat"],
                    lon=sl["lon"],
                )
                for ds in list_input
            ]
            input_altimeters_ds = xr.merge(list_input)
            item = (
                tgt_altimeter_ds.sel(
                    time=slice(ds_bools_patch["time"].values[0], ds_bools_patch["time"].values[-1]),
                    lat=sl["lat"],
                    lon=sl["lon"],
                )
                .to_dataset(name="tgt")
                .assign(input=input_altimeters_ds.to_dataarray().mean(dim="variable"))
                .to_array()
                .sortby('variable')
            )

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.noise is not None:
            noise = np.tile(
                self._rng.uniform(-self.noise, self.noise, item[0].shape), (2, 1, 1, 1)
            ).astype(np.float32)
            item = item + noise

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

def load_ose_data_on_fly(
):
    # TODO: change variables as parameters
    altimeters_name_list = ["al", "alg", "c2", "enn", "h2a", "h2ag", "j1g", "j1n", "j2", "j2g", "j2n", "j3", "s3a", "s3b"]
    altimeters_path_list = ["/Odyssey/public/altimetry_traces/2010_2019/gridded/ose_" + name + "_daily_sla_0.25deg.nc" for name in altimeters_name_list]
    isel = None
    ds = [
            xr.open_dataset(altimeters_path_list[i])[altimeters_name_list[i]]
            .isel(isel)
            .rename(latitude="lat", longitude="lon")
    for i in range(len(altimeters_path_list))]
    return ds

if __name__ == "__main__":
    ds = load_ose_data_on_fly()
    print(ds)

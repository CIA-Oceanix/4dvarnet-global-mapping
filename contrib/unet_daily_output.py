"""Daily supervised targets with sixty six-hour observation channels."""
import numpy as np
import xarray as xr
from contrib.glorys12 import DistinctNormDataModulewInputTracks

def load_daily_targets(tgt_path, inp_path, tgt_var='tgt', inp_var='obs'):
    daily=xr.open_dataset(tgt_path)[tgt_var]
    obs=xr.open_dataset(inp_path)[inp_var]
    np.testing.assert_array_equal(daily.lat,obs.lat)
    np.testing.assert_array_equal(daily.lon,obs.lon)
    t=obs.time.values
    assert np.all(np.diff(t)==np.timedelta64(6,'h'))
    assert t[0]==t[0].astype('datetime64[D]')
    np.testing.assert_array_equal(daily.time.sel(time=t[::4]),t[::4])
    # Lazy indexing only: no new dataset. Postprocessing selects exact daily
    # timestamps 0,4,...,56 before supervision; intermediate targets are unused.
    return daily.reindex(time=obs.time,method='ffill'),obs

class DailyTargetDataModule(DistinctNormDataModulewInputTracks):
    def setup(self,stage='fit'):
        for phase in ['train','val']:
            kw=self.xrds_kw[phase]
            assert kw['patch_dims']['time']==60
            assert kw['strides']['time']%4==0
            t=self.inp.sel(self.domains[phase]).time.values
            assert t[0]==t[0].astype('datetime64[D]')
        super().setup(stage)

    def post_fn(self,phase):
        normalize=super().post_fn(phase)
        def daily(item):
            item=normalize(item)
            assert item.input.shape[0]==60 and item.tgt.shape[0]==60
            return item._replace(tgt=item.tgt[::4])
        return daily

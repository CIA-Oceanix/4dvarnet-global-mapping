"""Stream linear 6-hour GLORYS targets and re-sampled synthetic track observations.

Uses only coordinates/times from existing yearly track caches, never their SLA values.
No extrapolation beyond the daily source; final year ends at the last source time.
All execution must be on a Slurm compute node.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

SOURCE = Path('/Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_8th_gridded_from_alongtrack_float32.nc')
CACHE = Path('/Odyssey/public/_transit_zone_/process_glorys/work_8th_p23leflo')
OUT = Path('/Odyssey/private/p23leflo/data/glorys8th_6h_linear_v2')
H6 = np.timedelta64(6, 'h')

def blend(a, b, w):
    if w == 0: return a.copy()
    if w == 1: return b.copy()
    return ((1-w)*a + w*b).astype('float32')

def create(path, times, lat, lon):
    nc = Dataset(path, 'w', format='NETCDF4')
    for k, n in [('time',len(times)), ('lat',len(lat)), ('lon',len(lon))]:
        nc.createDimension(k,n)
    t = nc.createVariable('time','i8',('time',))
    t.units='hours since 1970-01-01'; t.calendar='proleptic_gregorian'
    t[:] = times.astype('datetime64[h]').astype('int64')
    for k, a in [('lat',lat),('lon',lon)]:
        v=nc.createVariable(k,'f4',(k,)); v[:]=a
        v.units='degrees_north' if k=='lat' else 'degrees_east'
    for k in ['tgt','obs']:
        nc.createVariable(k,'f4',('time','lat','lon'),fill_value=np.nan,
                          zlib=True,complevel=1,chunksizes=(1,128,128)).units='m'
    nc.source_daily=str(SOURCE)
    nc.method='Linear temporal target interpolation; bilinear space/linear time track sampling; nearest cell and centered 6h bins'
    return nc

def build(year, smoke=False):
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT / (f'smoke_{year}.nc' if smoke else f'year_{year}.nc')
    if dest.exists(): raise FileExistsError(dest)
    with xr.open_dataset(SOURCE) as src:
        st=src.time.values.astype('datetime64[ns]')
        assert np.all(np.diff(st)==np.timedelta64(1,'D'))
        lat=src.lat.values; lon=src.lon.values
        assert np.allclose(np.diff(lat),.125) and np.allclose(np.diff(lon),.125)
        lo=max(st[0],np.datetime64(f'{year}-01-01','ns'))
        hi=min(st[-1],np.datetime64(f'{year+1}-01-01','ns')-H6)
        times=np.arange(lo,hi+H6,H6)
        if smoke: times=times[:8]
        # Load coordinates only, including adjacent year halos for midnight bins.
        ts=[]; ys=[]; xs=[]
        for y in range(max(2010,year-1),min(2019,year+1)+1):
            with xr.open_dataset(CACHE/f'interpolated_{y}.nc') as tr:
                t=tr.time.values
                ids=np.flatnonzero((t>=times[0]-H6/2)&(t<times[-1]+H6/2))
                ts.append(t[ids]); ys.append(tr.latitude.isel(time=ids).values)
                xs.append(tr.longitude.isel(time=ids).values)
        tt=np.concatenate(ts); yy=np.concatenate(ys); xx=np.concatenate(xs)
        order=np.argsort(tt); tt=tt[order]; yy=yy[order]; xx=xx[order]
        fields={}
        def field(i):
            if i not in fields: fields[i]=src.tgt.isel(time=i).values.astype('float32')
            return fields[i]
        def sample(t,y,x):
            values=np.full(len(t),np.nan)
            ix=np.searchsorted(st,t,side='right')-1
            for j in np.unique(ix):
                if j<0 or j>=len(st)-1: continue
                m=ix==j
                pts=np.column_stack((y[m],(x[m]-lon[0])%360+lon[0]))
                vals=[]
                for z in [j,j+1]:
                    f=field(int(z)); periodic=np.concatenate((f,f[:,:1]),axis=1)
                    vals.append(RegularGridInterpolator((lat,np.r_[lon,lon[0]+360]),periodic,
                                bounds_error=False,fill_value=np.nan)(pts))
                w=(t[m]-st[j])/np.timedelta64(1,'D')
                values[m]=(1-w)*vals[0]+w*vals[1]
            return values
        partial=Path(str(dest)+'.partial')
        if partial.exists(): raise FileExistsError(partial)
        with create(partial,times,lat,lon) as nc:
            for k,t in enumerate(times):
                j=int(np.searchsorted(st,t,side='right')-1)
                w=float((t-st[j])/np.timedelta64(1,'D'))
                a=field(j); target=a.copy() if w==0 else blend(a,field(j+1),w)
                l,r=np.searchsorted(tt,[t-H6/2,t+H6/2])
                v=sample(tt[l:r],yy[l:r],xx[l:r])
                valid=np.isfinite(v)&np.isfinite(yy[l:r])&np.isfinite(xx[l:r])
                y=yy[l:r][valid]; x=xx[l:r][valid]; v=v[valid]
                ii=np.rint((y-lat[0])/.125).astype(int)
                jj=np.rint(((x-lon[0])%360)/.125).astype(int)%len(lon)
                ok=(ii>=0)&(ii<len(lat)); flat=ii[ok]*len(lon)+jj[ok]
                sums=np.bincount(flat,weights=v[ok],minlength=len(lat)*len(lon))
                counts=np.bincount(flat,minlength=len(sums)); obs=np.full(len(sums),np.nan,'float32')
                np.divide(sums,counts,out=obs,where=counts>0)
                nc['tgt'][k]=target; nc['obs'][k]=obs.reshape(len(lat),len(lon))
                fields={z:f for z,f in fields.items() if z>=j-1}
                if k%40==0: print(f'{year} {k+1}/{len(times)} {t} finite_obs={np.count_nonzero(counts)}',flush=True)
        partial.rename(dest)
    with xr.open_dataset(dest) as d, xr.open_dataset(SOURCE) as s:
        np.testing.assert_allclose(d.tgt.isel(time=0),s.tgt.sel(time=times[0]),equal_nan=True)
        a=s.tgt.sel(time=times[0]).values; b=s.tgt.sel(time=times[0]+np.timedelta64(1,'D')).values
        np.testing.assert_allclose(d.tgt.isel(time=1),blend(a,b,.25),equal_nan=True)
    print(f'VALIDATED {dest}',flush=True)

def merge():
    dest=OUT/'glorys12_2010_2019_6h_linear.nc'
    if dest.exists(): raise FileExistsError(dest)
    files=[OUT/f'year_{y}.nc' for y in range(2010,2020)]
    times=[]
    for f in files:
        with xr.open_dataset(f) as d: times.append(d.time.values)
    t=np.concatenate(times)
    assert len(t)==14605 and np.all(np.diff(t)==H6), (len(t),t[0],t[-1])
    with xr.open_dataset(files[0]) as d: lat=d.lat.values; lon=d.lon.values
    tmp=Path(str(dest)+'.partial')
    if tmp.exists(): raise FileExistsError(tmp)
    count=0; total=0.; sq=0.; offset=0
    with create(tmp,t,lat,lon) as out:
        for f in files:
            with xr.open_dataset(f) as d:
                np.testing.assert_array_equal(d.lat,lat); np.testing.assert_array_equal(d.lon,lon)
                for k in range(d.sizes['time']):
                    for v in ['tgt','obs']: out[v][offset+k]=d[v].isel(time=k).values
                    # Deterministic statistics over training only, every tenth day at all four hours.
                    if t[offset+k]<np.datetime64('2018-01-01') and (offset+k)//4%10==0:
                        a=d.tgt.isel(time=k).values.astype('float64'); a=a[np.isfinite(a)]
                        count+=a.size; total+=a.sum(); sq+=np.square(a).sum()
                offset+=d.sizes['time']; out.sync()
                print(f'merged {f.name} {offset}/{len(t)}',flush=True)
    mean=total/count; std=float(np.sqrt(sq/count-mean**2))
    tmp.rename(dest)
    (OUT/'norm_stats.json').write_text(json.dumps({'mean':mean,'std':std,'count':count,'sampling':'train 2010-2017, every tenth day, all four 6h steps'},indent=2))
    print(f'VALIDATED merged {dest} norm={mean},{std}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--year',type=int); p.add_argument('--smoke',action='store_true'); p.add_argument('--merge',action='store_true')
    args=p.parse_args()
    if args.year: build(args.year,args.smoke)
    elif args.merge: merge()
    else: p.error('--year or --merge required')

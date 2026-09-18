"""Production des donnees SSH OSSE au 1/8 deg GLOBAL : obs along-track grillees + verite.

Adaptation du script de Daniel Zhu (/Odyssey/public/_transit_zone_/process_glorys/
process_glorys12.py) qui a produit la famille 1/4 deg (glorys12_..._4th_*). Changements :
  - SOURCE = la verite SLA 1/8 deg existante (glorys12_2010_2019_daily_sla_8th_double.nc,
    j25lee) au lieu du zos natif 1/12 (absent du disque) : obs = verite echantillonnee le
    long des traces = coherence OSSE exacte, et le tgt est une simple decoupe (grille
    IDENTIQUE, aucune interpolation) ;
  - traces = 6 nadirs 2019 (hors Saral/AltiKa, all_tracks.nc), DECALEES d'annee en annee
    pour couvrir 2010-2019 (logique commentee du script d origine) ;
  - breakpoint() retire, boucle complete reactivee ;
  - sortie float32 (la famille 4th float64 pesait 67 Go ; ici ~109 Go en float32) ;
  - resumable : chaque etape saute si sa sortie existe deja ;
  - decoupe par ANNEE pour les jobs array SLURM (--year), puis --merge.

Usage :  python process_glorys12_8th.py --year 2019
         python process_glorys12_8th.py --merge
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import pyinterp

DC = "/Odyssey/private/p23leflo/projects/4dvarnet-global-mapping/evaluation"
if DC not in sys.path:
    sys.path.append(DC)
from src.mod_interp import run_interpolation  # noqa: E402

SRC = "/Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_8th_double.nc"
TRACKS = "/Odyssey/public/_transit_zone_/process_glorys/sources/all_tracks.nc"
WORK = Path("/Odyssey/public/_transit_zone_/process_glorys/work_8th_p23leflo")
FINAL = ("/Odyssey/public/glorys/reanalysis/"
         "glorys12_2010_2019_daily_sla_8th_gridded_from_alongtrack_float32.nc")

LATS = np.arange(-80, 82, 1 / 8)
LONS = np.arange(-180, 180, 1 / 8)


def interp_year(year: int):
    """Interpole la verite 1/8 le long des traces 2019 decalees vers `year`."""
    out = WORK / f"interpolated_{year}.nc"
    if out.exists():
        print(f"[interp {year}] deja fait ({out})")
        return
    (WORK).mkdir(parents=True, exist_ok=True)

    print(f"[interp {year}] chargement des traces...")
    track = (
        xr.open_dataset(TRACKS)
        .sel(time="2019")
        .sortby("time")
    )
    offset = 2019 - year
    if offset:
        track = track.assign(time=track.time.to_index() - pd.DateOffset(years=offset))

    print(f"[interp {year}] chargement de la verite (annee {year})...")
    ds_y = xr.open_dataset(SRC).sel(time=str(year))[["sla"]].load()

    print(f"[interp {year}] interpolation le long des traces...")
    run_interpolation(ds_y, track).to_netcdf(out)
    print(f"[interp {year}] OK -> {out}")


def grid_year(year: int):
    """Grille les obs along-track au 1/8 (binning journalier) + verite decoupee."""
    out = WORK / f"gridded_{year}.nc"
    if out.exists():
        print(f"[grid {year}] deja fait ({out})")
        return

    print(f"[grid {year}] verite (decoupe exacte, pas d interpolation)...")
    tgt_y = (
        xr.open_dataset(SRC)
        .sel(time=str(year), latitude=slice(LATS[0] - 1e-6, LATS[-1] + 1e-6))
        .sla.load()
    )
    assert np.allclose(tgt_y.latitude.values, LATS), "grille lat inattendue"
    assert np.allclose(tgt_y.longitude.values, LONS), "grille lon inattendue"

    obs_ds = (
        xr.open_dataset(WORK / f"interpolated_{year}.nc")
        .rename(msla_interpolated="sla")
        .assign(longitude=lambda x: (x.longitude - 180) % 360 - 180)
        .sortby("longitude")
        [["latitude", "longitude", "sla"]]
        .load()
    )

    times = xr.Dataset(
        coords=dict(time=pd.date_range(f"{year}-01-01", f"{year}-12-31"))
    ).time
    binning = pyinterp.Binning2D(pyinterp.Axis(LONS), pyinterp.Axis(LATS))

    def grid_da(da):
        binning.clear()
        values = np.ravel(da["sla"].values)
        lons = np.ravel(da.longitude.values)
        lats = np.ravel(da.latitude.values)
        msk = np.isfinite(values)
        binning.push(lons[msk], lats[msk], values[msk])
        return (("time", "latitude", "longitude"),
                binning.variable("mean").T[None, ...].astype(np.float32))

    gridded = []
    t_res = times.diff("time").values.mean()
    for t in tqdm.tqdm(times):
        tds = obs_ds.isel(
            time=(obs_ds.time > (t - t_res / 2)) & (obs_ds.time <= (t + t_res / 2))
        )
        gridded.append(
            xr.Dataset(
                data_vars={
                    "tgt": tgt_y.sel(time=t.dt.strftime("%Y-%m-%d").item()).astype(np.float32),
                    "obs": grid_da(tds),
                },
                coords={
                    "time": [t.values],
                    "latitude": np.array(binning.y, dtype=np.float32),
                    "longitude": np.array(binning.x, dtype=np.float32),
                },
            )
        )
    final = xr.concat(gridded, dim="time").sortby("time")
    final.to_netcdf(out)
    print(f"[grid {year}] OK -> {out}")


def merge():
    """Concatene les 10 annees -> fichier final (lat/lon, float32)."""
    if Path(FINAL).exists():
        print(f"[merge] deja fait ({FINAL})")
        return
    missing = [y for y in range(2010, 2020) if not (WORK / f"gridded_{y}.nc").exists()]
    assert not missing, f"annees manquantes : {missing}"
    print("[merge] concatenation 2010-2019...")
    ds = xr.open_mfdataset(
        [str(WORK / f"gridded_{y}.nc") for y in range(2010, 2020)],
        combine="nested", concat_dim="time",
    ).sortby("time").rename(latitude="lat", longitude="lon")
    ds.to_netcdf(FINAL)
    print(f"[merge] OK -> {FINAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="traite une annee (interp + grillage)")
    ap.add_argument("--merge", action="store_true", help="fusionne 2010-2019")
    args = ap.parse_args()
    if args.year:
        assert 2010 <= args.year <= 2019
        interp_year(args.year)
        grid_year(args.year)
    if args.merge:
        merge()
    if not args.year and not args.merge:
        ap.error("--year N et/ou --merge requis")

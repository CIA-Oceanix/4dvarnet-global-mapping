"""Entree OSE au 1/8 de degre, analogue exact de sla_filtered_0.25deg.nc.

Recette validee contre la cible au quart (0 maille manquante, 99,94 % bit-a-bit) :
  - fenetre centree sur le jour : [t0 - 12 h, t0 + 12 h[
  - arrondi au centre de maille le plus proche
  - emission sur toutes les colonnes congruentes mod 360 : la grille couvre
    372 deg, soit 12 deg de recouvrement au-dela du tour complet
  - moyenne arithmetique des observations d'une meme maille

Le fichier au quart contient 5 nadirs (c2, h2ag, j3, s3a, s3b), plus h2b a partir
du 20 decembre 2019. On garde la meme constellation.
"""
import glob, sys, numpy as np, xarray as xr
from netCDF4 import Dataset

AT   = "/Odyssey/public/data_challenge_ssh_ose/data/alongtrack"
G    = "/Odyssey/public/data_challenge_ssh_ose/data/gridded_alongtrack"
OUT  = "/Odyssey/private/p23leflo/data/ose_8th/sla_filtered_0.125deg.nc"
SATS = ("c2", "h2ag", "h2b", "j3", "s3a", "s3b")
T0   = np.datetime64("2018-12-01"); NT = 427
RES  = 0.125
LAT  = np.arange(-79.9375, 81.9375 + 1e-9, RES)
LON  = np.arange(-185.9375, 185.9375 + 1e-9, RES)
NLA, NLO = len(LAT), len(LON)

def collecte(t0):
    """Points (lat, lon, sla_filtered) de la fenetre centree sur t0."""
    tlo, thi = t0 - np.timedelta64(12,"h"), t0 + np.timedelta64(12,"h")
    jours = [str((t0+np.timedelta64(k,"D")).astype("datetime64[D]")).replace("-","")
             for k in (-1, 0, 1)]
    LA, LO, SL = [], [], []
    for j in jours:
        for s in SATS:
            for f in glob.glob(f"{AT}/{s}/*/*_{j}_*.nc"):
                with xr.open_dataset(f) as d:
                    t = d["time"].values; k = (t >= tlo) & (t < thi)
                    if not k.any(): continue
                    la, lo = d["latitude"].values[k], d["longitude"].values[k]
                    sl = d["sla_filtered"].values[k]
                    m = np.isfinite(la) & np.isfinite(lo) & np.isfinite(sl)
                    LA.append(la[m]); LO.append(lo[m]); SL.append(sl[m])
    if not LA: return None
    return np.concatenate(LA), np.concatenate(LO), np.concatenate(SL)

def grille(la, lo, sl, lat_ax, lon_ax, res):
    nla, nlo = len(lat_ax), len(lon_ax)
    i = np.rint((la - lat_ax[0]) / res).astype(np.int64)
    II, JJ, VV = [], [], []
    for shift in (-360.0, 0.0, 360.0):          # colonnes congruentes mod 360
        j = np.rint(((lo + shift) - lon_ax[0]) / res).astype(np.int64)
        v = (i >= 0) & (i < nla) & (j >= 0) & (j < nlo)
        II.append(i[v]); JJ.append(j[v]); VV.append(sl[v])
    f = np.concatenate(II) * nlo + np.concatenate(JJ)
    val = np.concatenate(VV).astype("float64")
    num = np.bincount(f, weights=val, minlength=nla*nlo)
    cnt = np.bincount(f, minlength=nla*nlo)
    g = np.full(nla*nlo, np.nan); nz = cnt > 0; g[nz] = num[nz]/cnt[nz]
    return g.reshape(nla, nlo).astype("float32")

# ---- validation : rejouer au quart contre la cible -------------------------
ref = xr.open_dataset(f"{G}/sla_filtered_0.25deg.nc")
la4, lo4 = ref["lat"].values, ref["lon"].values
print("### validation au quart (0 maille manquante attendue)", flush=True)
for D in ("2018-12-15", "2019-05-05", "2019-12-05", "2020-01-15"):
    t0 = np.datetime64(D); c = collecte(t0)
    g = grille(*c, la4, lo4, 0.25)
    r = ref["sla_filtered"].sel(time=D).values.astype("float32")
    a, b = np.isfinite(g), np.isfinite(r); both = a & b
    dif = np.abs(g[both] - r[both])
    print(f"  {D}  nous {int(a.sum()):6d}  cible {int(b.sum()):6d}  "
          f"cible seule {int((b&~a).sum()):4d}  nous seuls {int((a&~b).sum()):4d}  "
          f"identiques {100*np.mean(dif==0):.3f} %", flush=True)
ref.close()

# ---- production au 1/8 -----------------------------------------------------
print(f"\n### production {NT} jours sur {NLA} x {NLO}  ->  {OUT}", flush=True)
nc = Dataset(OUT, "w", format="NETCDF4")
nc.createDimension("time", NT); nc.createDimension("lat", NLA); nc.createDimension("lon", NLO)
vt = nc.createVariable("time", "i8", ("time",))
vt.units = "days since 2018-12-01"; vt.calendar = "proleptic_gregorian"
vla = nc.createVariable("lat", "f8", ("lat",)); vla.units = "degrees_north"
vlo = nc.createVariable("lon", "f8", ("lon",)); vlo.units = "degrees_east"
vs = nc.createVariable("sla_filtered", "f4", ("time","lat","lon"),
                       fill_value=np.float32(np.nan), contiguous=False, zlib=False)
vs.units = "m"
vs.long_name = ("Sea level anomaly filtered not-subsampled with dac, ocean_tide "
                "and lwe correction applied, gridded at 1/8 degree")
vla[:] = LAT; vlo[:] = LON; vt[:] = np.arange(NT)
tot = 0
for k in range(NT):
    t0 = T0 + np.timedelta64(k, "D")
    c = collecte(t0)
    g = (np.full((NLA, NLO), np.nan, "float32") if c is None
         else grille(*c, LAT, LON, RES))
    vs[k, :, :] = g
    n = int(np.isfinite(g).sum()); tot += n
    if k % 25 == 0 or k == NT-1:
        print(f"  {str(t0)[:10]}  jour {k+1:3d}/{NT}  {n:7d} mailles", flush=True)
nc.close()
print(f"\n  moyenne : {tot/NT:.0f} mailles observees par jour", flush=True)
print("[fin]", flush=True)

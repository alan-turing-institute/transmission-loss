"""
propagation.py — Analytic sound-speed profile and transmission-loss model.

The fast, closed-form model whose error Bellhop is used to correct. Two parts:

  Sound speed  : WOA23 monthly T/S climatology -> Mackenzie (1981) profile ->
                 per-layer mean sound speed.
  Analytic TL  : spherical/cylindrical spreading + Thorp (1967) absorption +
                 geometric shadow penalty + per-layer correction.

`analytic_tl_point(...)` returns the analytic TL in dB for one source→receiver
geometry; `shadow_obstructed(...)` decides whether the straight source→surface
line dips below the seabed (triggering the shadow penalty).

In the pipeline: `generate_dataset.py` calls `analytic_tl_point` for the analytic
TL of each geometry, and passes the sound-speed profile from `sound_speed_profile`
on to Bellhop so both models see the same water column.

Inputs
------
WOA23 CSV.gz climatology directories, e.g.
    Data/woa23_t_B5C2_1.00_csv/  (temperature)
    Data/woa23_s_B5C2_1.00_csv/  (salinity)
"""
from __future__ import annotations

import gzip
import math
import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ── Analytic-model parameters ─────────────────────────────────────────────────
R_TRANSITION_KM: float = 45.0    # spherical -> cylindrical spreading transition
SHADOW_PENALTY_DB: float = 30.0  # TL added for geometrically obstructed paths
SAMPLE_INTERVAL_KM: float = 1.0  # great-circle shadow sampling step (km)

LAYER_INTERFACES_M: list[float] = [200.0, 600.0]
CLEARANCE_M: float = 50.0

# Empirical per-layer corrections relative to surface (dB); negative = less loss.
LAYER_TL_CORRECTION_DB: dict[str, float] = {"surface": 0.0, "mid": -3.0, "deep": -8.0}

_R_EARTH_M: float = 6_371_000.0

# 57 standard WOA depth levels (m)
WOA_DEPTHS_M: np.ndarray = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
    90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400,
    425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
    1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
], dtype=np.float64)
_N_WOA = len(WOA_DEPTHS_M)


# ── WOA loading ───────────────────────────────────────────────────────────────

def _iter_rows(path: str, lat_min: float, lat_max: float,
               lon_min: float, lon_max: float):
    """Yield (lat, lon, vals) for in-domain rows of a WOA CSV.gz file."""
    with gzip.open(path, "rt") as f:
        next(f); next(f)  # skip description + depth-header lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                lat = float(parts[0]); lon = float(parts[1])
            except (ValueError, IndexError):
                continue
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
            vals = np.full(_N_WOA, np.nan, dtype=np.float32)
            for k in range(min(len(parts) - 2, _N_WOA)):
                try:
                    vals[k] = float(parts[k + 2])
                except ValueError:
                    pass
            yield lat, lon, vals


def load_woa(
    temp_dir: str, sal_dir: str,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
) -> xr.Dataset:
    """
    Load WOA23 monthly climatological temperature and salinity for the domain.

    Returns an xr.Dataset with variables 'temperature' and 'salinity', dims
    (month, lat, lon, depth); month = 1..12; missing depths are NaN.
    """
    path01 = os.path.join(temp_dir, "woa23_B5C2_t01an01.csv.gz")
    coords = {(lat, lon) for lat, lon, _ in
              _iter_rows(path01, lat_min, lat_max, lon_min, lon_max)}
    grid_lats = sorted({c[0] for c in coords})
    grid_lons = sorted({c[1] for c in coords})
    lat_to_i = {lat: i for i, lat in enumerate(grid_lats)}
    lon_to_j = {lon: j for j, lon in enumerate(grid_lons)}
    n_lat, n_lon = len(grid_lats), len(grid_lons)

    temp = np.full((12, n_lat, n_lon, _N_WOA), np.nan, dtype=np.float32)
    sal = np.full((12, n_lat, n_lon, _N_WOA), np.nan, dtype=np.float32)
    for mm in range(1, 13):
        for d, arr in [(temp_dir, temp), (sal_dir, sal)]:
            v = "t" if arr is temp else "s"
            path = os.path.join(d, f"woa23_B5C2_{v}{mm:02d}an01.csv.gz")
            for lat, lon, vals in _iter_rows(path, lat_min, lat_max, lon_min, lon_max):
                i, j = lat_to_i.get(lat), lon_to_j.get(lon)
                if i is not None and j is not None:
                    arr[mm - 1, i, j, :] = vals

    return xr.Dataset(
        {"temperature": (["month", "lat", "lon", "depth"], temp),
         "salinity":    (["month", "lat", "lon", "depth"], sal)},
        coords={"month": np.arange(1, 13, dtype=np.int32),
                "lat": np.array(grid_lats), "lon": np.array(grid_lons),
                "depth": WOA_DEPTHS_M},
    )


# ── Sound-speed profile (Mackenzie 1981) ──────────────────────────────────────

def sound_speed_profile(
    woa: xr.Dataset, lat: float, lon: float, month: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mackenzie (1981) sound speed at each WOA depth level (nearest-neighbour lookup).

    Returns
    -------
    ssp      : (n_depth,) sound speed m/s; NaN where T or S is missing.
    depths_m : (n_depth,) WOA depth levels in metres.
    """
    T = (woa["temperature"].sel(lat=lat, lon=lon, month=month, method="nearest")
         .values.astype(np.float64))
    S = (woa["salinity"].sel(lat=lat, lon=lon, month=month, method="nearest")
         .values.astype(np.float64))
    z = WOA_DEPTHS_M
    c = (1448.96
         + 4.591 * T - 0.05304 * T ** 2 + 0.0002374 * T ** 3
         + 0.01630 * z
         + (1.340 - 0.01025 * T) * (S - 35.0)
         + 1.675e-7 * z ** 2 - 7.139e-13 * T * z ** 3)
    return c, z.copy()


def layer_mean_speed(
    ssp: np.ndarray, depths_m: np.ndarray, layer: str,
    node_depth_m: float,
    layer_interfaces_m: list[float] = LAYER_INTERFACES_M,
    clearance_m: float = CLEARANCE_M,
) -> float:
    """
    Mean sound speed (m/s) over the depth band of `layer` at this node.

    Trapezoidal integration; band edges falling between WOA levels are linearly
    interpolated. Returns NaN if the layer is not present or has no valid data.
    """
    iface0, iface1 = layer_interfaces_m
    max_depth = node_depth_m - clearance_m
    if layer == "surface":
        z_lo, z_hi = 0.0, min(iface0, max_depth)
    elif layer == "mid":
        z_lo, z_hi = iface0, min(iface1, max_depth)
    elif layer == "deep":
        z_lo, z_hi = iface1, max_depth
    else:
        raise ValueError(f"Unknown layer: {layer!r}")
    if z_hi <= z_lo + 1e-6:
        return float("nan")

    valid = ~np.isnan(ssp)
    if not valid.any():
        return float("nan")
    z_v, c_v = depths_m[valid], ssp[valid]

    in_band = (z_v >= z_lo) & (z_v <= z_hi)
    z_b, c_b = z_v[in_band].tolist(), c_v[in_band].tolist()

    def _interp(z_target: float) -> float | None:
        if z_target < z_v[0] or z_target > z_v[-1]:
            return None
        return float(np.interp(z_target, z_v, c_v))

    if not z_b or z_b[0] > z_lo + 1e-6:
        c_lo = _interp(z_lo)
        if c_lo is not None:
            z_b.insert(0, z_lo); c_b.insert(0, c_lo)
    if not z_b or z_b[-1] < z_hi - 1e-6:
        c_hi = _interp(z_hi)
        if c_hi is not None:
            z_b.append(z_hi); c_b.append(c_hi)
    if len(z_b) < 2:
        return float("nan")

    z_arr, c_arr = np.array(z_b), np.array(c_b)
    integral = float(np.sum(0.5 * (c_arr[:-1] + c_arr[1:]) * np.diff(z_arr)))
    return integral / (z_arr[-1] - z_arr[0])


# ── Absorption and shadow detection ───────────────────────────────────────────

def thorp_absorption(f_hz: float) -> float:
    """Thorp (1967) absorption coefficient in dB/km for frequency f_hz."""
    f = f_hz / 1000.0
    f2 = f * f
    return (0.011 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2)
            + 2.75e-4 * f2 + 0.003)


def shadow_obstructed(
    source_lat: float, source_lon: float, source_depth_m: float,
    receiver_lat: float, receiver_lon: float,
    rgi: RegularGridInterpolator,
    interval_km: float = SAMPLE_INTERVAL_KM,
) -> bool:
    """
    True if the straight line in (range, depth) from source to the surface receiver
    passes below the seabed at any point sampled every `interval_km` along the
    great circle. `rgi` returns seabed depth (m); receiver assumed at 0 m.
    """
    dlat = math.radians(receiver_lat - source_lat)
    dlon = math.radians(receiver_lon - source_lon)
    phi1, phi2 = math.radians(source_lat), math.radians(receiver_lat)
    a = math.sin(dlat / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    range_km = 2.0 * _R_EARTH_M * math.asin(math.sqrt(max(0.0, min(1.0, a)))) / 1000.0
    if range_km < interval_km:
        return False

    lat_s, lon_s = math.radians(source_lat), math.radians(source_lon)
    lat_r, lon_r = math.radians(receiver_lat), math.radians(receiver_lon)
    x1, y1, z1 = math.cos(lat_s) * math.cos(lon_s), math.cos(lat_s) * math.sin(lon_s), math.sin(lat_s)
    x2, y2, z2 = math.cos(lat_r) * math.cos(lon_r), math.cos(lat_r) * math.sin(lon_r), math.sin(lat_r)
    omega = math.acos(max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2)))
    sin_om = math.sin(omega)

    for k in range(1, int(range_km / interval_km) + 1):
        frac = k * interval_km / range_km
        if frac >= 1.0:
            break
        if sin_om < 1e-10:
            lat_i, lon_i = source_lat, source_lon
        else:
            a_c = math.sin((1.0 - frac) * omega) / sin_om
            b_c = math.sin(frac * omega) / sin_om
            xi, yi, zi = a_c * x1 + b_c * x2, a_c * y1 + b_c * y2, a_c * z1 + b_c * z2
            lat_i = math.degrees(math.atan2(zi, math.sqrt(xi ** 2 + yi ** 2)))
            lon_i = math.degrees(math.atan2(yi, xi))
        ray_depth = source_depth_m * (1.0 - frac)  # linear from src depth to 0
        bathy_val = float(rgi([[lat_i, lon_i]])[0])
        if math.isnan(bathy_val) or ray_depth > bathy_val:
            return True
    return False


# ── Analytic transmission loss ────────────────────────────────────────────────

def analytic_tl_point(
    range_km: float, freq_hz: float, layer: str, shadow_flag: bool,
) -> float:
    """
    Analytic TL in dB for one source→receiver geometry.

    Spherical spreading (20 log10 r_m) transitions to cylindrical
    (+10 log10 r/r_t) at r_t = 45 km, plus Thorp absorption, per-layer
    correction, and a fixed shadow penalty when the path is obstructed.
    """
    r_t = R_TRANSITION_KM
    r_m = range_km * 1000.0
    if range_km <= r_t:
        tl_spread = 20.0 * math.log10(max(r_m, 1.0))
    else:
        tl_spread = 20.0 * math.log10(r_t * 1000.0) + 10.0 * math.log10(range_km / r_t)
    absorption = thorp_absorption(freq_hz) * range_km
    layer_corr = LAYER_TL_CORRECTION_DB.get(layer, 0.0)
    shadow_add = SHADOW_PENALTY_DB if shadow_flag else 0.0
    return tl_spread + absorption + layer_corr + shadow_add


# ── Test / usage example ──────────────────────────────────────────────────────
# Inputs : WOA23 dirs, a point in the Denmark Strait, month = Jan.
# Outputs: surface sound speed, layer-mean speeds, and analytic TL at 4 freqs.
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.normpath(os.path.join(HERE, "Data"))
    TEMP_DIR = os.path.join(DATA_ROOT, "woa23_t_B5C2_1.00_csv")
    SAL_DIR = os.path.join(DATA_ROOT, "woa23_s_B5C2_1.00_csv")

    woa = load_woa(TEMP_DIR, SAL_DIR, 62.0, 70.0, -44.0, -13.0)
    lat, lon, month = 65.5, -25.5, 1
    c, z = sound_speed_profile(woa, lat, lon, month)
    print(f"Surface sound speed at {lat} N {abs(lon)} W (Jan): {c[0]:.1f} m/s")
    for layer in ("surface", "mid", "deep"):
        print(f"  {layer:7s} mean speed: "
              f"{layer_mean_speed(c, z, layer, 1500.0):.1f} m/s")

    print("Analytic TL (surface, 120 km, no shadow):")
    for f in (100.0, 300.0, 1000.0, 3000.0):
        print(f"  {f:6.0f} Hz -> {analytic_tl_point(120.0, f, 'surface', False):.1f} dB")

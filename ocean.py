"""
ocean.py — Bathymetry, ocean nodes, and great-circle geometry.

Shared infrastructure for the TL-correction minimal example. Two responsibilities:

1. Turn a GEBCO bathymetry file into a list of `OceanNode`s on a ~5 km grid, each
   carrying its seabed depth and the source depths of the acoustic layers
   (surface / mid / deep) that are physically available there.
2. Great-circle geometry helpers used to measure ranges and to sample the seabed
   profile along a source→receiver path (needed by both the analytic model and
   Bellhop).

Every ocean cell is a candidate source or receiver — for TL data we only need
diverse ocean-to-ocean pairs.

In the pipeline: `generate_dataset.py` builds the ocean nodes, samples random
pairs from them, and uses the geometry helpers here to profile the seabed along
each source→receiver path before handing that path to Bellhop.

Inputs
------
GEBCO netCDF with an `elevation` variable (negative = below sea level), e.g.
    Data/GEBCO_01_Jun_2026_.../gebco_2026_n72.0_s62.0_w-45.0_e-10.0.nc

Outputs
-------
`load_bathymetry(...)`  -> xr.DataArray depth (m, positive down), dims (lat, lon)
`build_ocean_nodes(...)` -> list[OceanNode]
`gebco_rgi(...)`         -> RegularGridInterpolator over full-resolution GEBCO depth
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ── Physical / grid parameters (shared across scenarios) ──────────────────────
R_EARTH_M: float = 6_371_000.0
GRID_RESOLUTION_KM: float = 5.0

LAYER_INTERFACES_M: list[float] = [200.0, 600.0]  # surface/mid, mid/deep boundaries
CLEARANCE_M: float = 50.0                          # min source clearance above seabed
LAYERS: list[str] = ["surface", "mid", "deep"]


# ── Great-circle geometry ─────────────────────────────────────────────────────

def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (Haversine) between two lat/lon points (deg)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    return 2.0 * R_EARTH_M * math.asin(math.sqrt(a)) / 1000.0


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (deg, 0 = north, 90 = east) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def point_at_bearing(
    lat0: float, lon0: float, bearing_deg: float, range_km: float
) -> tuple[float, float]:
    """(lat, lon) in degrees at `range_km` from (lat0, lon0) along `bearing_deg`."""
    d = range_km / 6371.0
    b = math.radians(bearing_deg)
    ph0 = math.radians(lat0)
    la0 = math.radians(lon0)
    phi = math.asin(math.sin(ph0) * math.cos(d) + math.cos(ph0) * math.sin(d) * math.cos(b))
    lam = la0 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(ph0),
        math.cos(d) - math.sin(ph0) * math.sin(phi),
    )
    return math.degrees(phi), math.degrees(lam)


def sample_path_bathy(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    step_km: float,
    rgi: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seabed depth sampled every `step_km` along the great circle from point 1 to 2.

    rgi : interpolator returning seabed depth (m, positive down) at (lat, lon);
          NaN over land / outside the grid.

    Returns
    -------
    ranges_km : (n,) distance from source in km
    depths_m  : (n,) seabed depth in m (positive down); NaN = land / out of domain
    """
    range_km = great_circle_km(lat1, lon1, lat2, lon2)
    brg = bearing_between(lat1, lon1, lat2, lon2)
    ranges = np.arange(0.0, range_km + step_km * 0.5, step_km)
    pts = np.array([list(point_at_bearing(lat1, lon1, brg, float(r))) for r in ranges])
    return ranges, rgi(pts)


# ── Bathymetry loading ────────────────────────────────────────────────────────

def load_bathymetry(
    nc_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_km: float = GRID_RESOLUTION_KM,
) -> xr.DataArray:
    """
    Load GEBCO bathymetry, clip to the bounding box, and regrid to ~resolution_km.

    Returns seabed depth in metres (positive downward); NaN for land cells.
    Dims: ('lat', 'lon').
    """
    ds = xr.open_dataset(nc_path)
    raw: xr.DataArray = ds["elevation"]

    buf = 0.2  # small buffer so interpolation has support at the domain edges
    raw = raw.sel(
        lat=slice(lat_min - buf, lat_max + buf),
        lon=slice(lon_min - buf, lon_max + buf),
    )

    center_lat = (lat_min + lat_max) / 2.0
    dlat = resolution_km / 111.195
    dlon = resolution_km / (111.195 * math.cos(math.radians(center_lat)))

    target_lats = np.arange(lat_min, lat_max + dlat * 0.5, dlat)
    target_lons = np.arange(lon_min, lon_max + dlon * 0.5, dlon)

    regridded = raw.interp(lat=target_lats, lon=target_lons, method="linear")
    depth = xr.where(regridded < 0.0, -regridded, np.nan)
    depth.attrs.update(long_name="seabed depth", units="m")
    return depth


def gebco_rgi(nc_path: str) -> RegularGridInterpolator:
    """
    Full-resolution GEBCO depth interpolator for sampling the seabed profile along
    a Bellhop path. Depth is positive down; NaN over land / outside the file.
    """
    ds = xr.open_dataset(nc_path)
    depth = xr.where(ds["elevation"] < 0.0, -ds["elevation"], np.nan)
    return RegularGridInterpolator(
        (depth.lat.values, depth.lon.values), depth.values,
        method="linear", bounds_error=False, fill_value=np.nan,
    )


def grid_rgi(bathy: xr.DataArray) -> RegularGridInterpolator:
    """Interpolator over the regridded 5 km depth grid (used for shadow checks)."""
    return RegularGridInterpolator(
        (bathy.lat.values, bathy.lon.values), bathy.values,
        method="linear", bounds_error=False, fill_value=np.nan,
    )


# ── Ocean nodes ───────────────────────────────────────────────────────────────

@dataclass
class OceanNode:
    lat: float
    lon: float
    depth_m: float                          # seabed depth, positive downward
    available_layers: list[str]             # subset of LAYERS present here
    layer_source_depth_m: dict[str, float]  # representative source depth per layer


def build_ocean_nodes(
    bathy: xr.DataArray,
    layer_interfaces_m: list[float] = LAYER_INTERFACES_M,
    clearance_m: float = CLEARANCE_M,
) -> list[OceanNode]:
    """
    One OceanNode per ocean cell of the regridded grid.

    Layer availability and representative source depths:
      surface : always            source depth 10 m
      mid     : seabed > 200+clr  source depth mean(200, 600)
      deep    : seabed > 600+clr  source depth mean(600, seabed-clr)
    """
    iface0, iface1 = layer_interfaces_m
    clr = clearance_m
    depth = bathy.values
    lats = bathy["lat"].values
    lons = bathy["lon"].values

    nodes: list[OceanNode] = []
    ocean_i, ocean_j = np.where(~np.isnan(depth) & (depth > 0.0))
    for i, j in zip(ocean_i, ocean_j):
        d = float(depth[i, j])

        avail = ["surface"]
        if d > iface0 + clr:
            avail.append("mid")
        if d > iface1 + clr:
            avail.append("deep")

        src = {"surface": 10.0}
        if "mid" in avail:
            src["mid"] = (iface0 + iface1) / 2.0
        if "deep" in avail:
            src["deep"] = (iface1 + d - clr) / 2.0

        nodes.append(OceanNode(
            lat=float(lats[i]), lon=float(lons[j]), depth_m=d,
            available_layers=avail, layer_source_depth_m=src,
        ))
    return nodes


# ── Test / usage example ──────────────────────────────────────────────────────
# Inputs : Denmark Strait GEBCO extract (62–70 N, 44–13 W)
# Outputs: ~thousands of ocean nodes; a sample node's depth/layers; a path profile.
if __name__ == "__main__":
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.normpath(os.path.join(HERE, "Data"))
    GEBCO = os.path.join(
        DATA_ROOT,
        "GEBCO_01_Jun_2026_cd11525db157",
        "gebco_2026_n72.0_s62.0_w-45.0_e-10.0.nc",
    )

    bathy = load_bathymetry(GEBCO, 62.0, 70.0, -44.0, -13.0)
    nodes = build_ocean_nodes(bathy)
    print(f"Grid {bathy.shape}  ->  {len(nodes)} ocean nodes")

    deep = [n for n in nodes if "deep" in n.available_layers]
    n = deep[len(deep) // 2]
    print(f"Sample node: {n.lat:.2f} N {abs(n.lon):.2f} W  seabed {n.depth_m:.0f} m")
    print(f"  layers {n.available_layers}  source depths {n.layer_source_depth_m}")

    rgi = gebco_rgi(GEBCO)
    r_km, dep = sample_path_bathy(n.lat, n.lon, nodes[0].lat, nodes[0].lon, 5.0, rgi)
    print(f"Path to node 0: {r_km[-1]:.0f} km, "
          f"seabed {np.nanmin(dep):.0f}-{np.nanmax(dep):.0f} m over {len(r_km)} samples")

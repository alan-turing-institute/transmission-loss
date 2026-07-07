"""
generate_dataset.py — Build the training dataset of analytic-vs-Bellhop TL.

This is the driver that ties the other modules together. For each range band and
each month it samples random ocean node pairs (`ocean`), then for every available
source layer and frequency computes two transmission-loss values for the same
geometry:

    tl_analytic_db  — the fast analytic model      (propagation.analytic_tl_point)
    tl_bellhop_db   — the reference ray trace       (bellhop.run_bellhop)

Both, plus the geometry/environment features that describe each pair, are written
as one row to a single CSV. `fit_xgb.py` then reads that CSV and learns to predict
the residual (tl_bellhop − tl_analytic), i.e. the correction the analytic model is
missing.

Pairs are drawn from three range bands, each with its own pair count and RNG seed,
and every band is evaluated across all 12 months. This spreads the dataset over a
wide range of distances and seasonal sound-speed conditions.

Inputs  : GEBCO extract + WOA23 dirs (paths below; the only files outside this folder).
Outputs : Data/BellhopData/bellhop_analytic.csv — one row per (pair, layer, frequency)
          Figures/dataset_range_hist.png      — range coverage per band
          Figures/dataset_residual_scatter.png — Bellhop vs analytic TL
"""
from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import ocean
import propagation as prop
from bellhop import prepare_ssp, run_bellhop

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(HERE, "Data"))
GEBCO_PATH = os.path.join(
    DATA_ROOT, "GEBCO_01_Jun_2026_cd11525db157",
    "gebco_2026_n72.0_s62.0_w-45.0_e-10.0.nc",
)
TEMP_DIR = os.path.join(DATA_ROOT, "woa23_t_B5C2_1.00_csv")
SAL_DIR = os.path.join(DATA_ROOT, "woa23_s_B5C2_1.00_csv")

OUT_DIR = os.path.join(DATA_ROOT, "BellhopData")
FIG_DIR = os.path.join(HERE, "Figures")
CSV_PATH = os.path.join(OUT_DIR, "bellhop_analytic.csv")

# ── Domain (Denmark Strait) ───────────────────────────────────────────────────
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 62.0, 70.0, -44.0, -13.0

# ── Sampling configuration ────────────────────────────────────────────────────
# Each band contributes pairs in its own distance range, so together they cover
# short to long ranges. Every band is run across all 12 months (below). Each band
# has a distinct RNG seed so the three sets of pairs are independent. Bellhop is
# the slow step — reduce n_pairs (or MONTHS) for a quick trial run.
RANGE_BANDS = [
    # name,    min_km, max_km, n_pairs, seed
    ("short",   10.0,  100.0,  200,     42),
    ("mid",    100.0,  200.0,   50,    200),
    ("long",   200.0,  500.0,   20,    400),
]
MONTHS = list(range(1, 13))
FREQS_HZ = [100.0, 300.0, 1000.0, 3000.0]
RECEIVER_DEPTH_M = 10.0
BATHY_STEP_KM = 2.0  # along-path seabed sampling for Bellhop

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FIELDNAMES = [
    "band", "month", "pair_i", "group_id",
    "src_lat", "src_lon", "rcv_lat", "rcv_lon", "range_km",
    "src_seabed_depth_m", "rcv_seabed_depth_m",
    "path_min_depth_m", "path_mean_depth_m",
    "layer", "src_depth_m", "rcv_depth_m", "freq_hz",
    "layer_mean_speed_ms", "shadow", "shadow_penalty_db",
    "tl_analytic_db", "tl_bellhop_db",
]


# ── Pair selection ────────────────────────────────────────────────────────────

def select_pairs(
    nodes: list[ocean.OceanNode],
    n_pairs: int, min_km: float, max_km: float,
    rng: np.random.Generator,
) -> list[tuple[int, int, float]]:
    """
    n_pairs (src_idx, rcv_idx, range_km) sampled from a large random candidate
    pool, stratified across two equal range sub-bins (split at the band midpoint)
    to spread ranges evenly. Remainder goes to the lower bin.
    """
    mid = (min_km + max_km) / 2.0
    edges = [min_km, mid, max_km]
    base = n_pairs // 2
    targets = [base + n_pairs % 2, base]

    n = len(nodes)
    pool = max(n_pairs * 200, 20_000)
    src = rng.integers(0, n, size=pool)
    rcv = rng.integers(0, n, size=pool)
    same = src == rcv
    rcv[same] = (rcv[same] + 1) % n

    lat = np.array([node.lat for node in nodes])
    lon = np.array([node.lon for node in nodes])
    ranges = ocean.R_EARTH_M / 1000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(
        np.sin(np.radians(lat[rcv] - lat[src]) / 2) ** 2
        + np.cos(np.radians(lat[src])) * np.cos(np.radians(lat[rcv]))
        * np.sin(np.radians(lon[rcv] - lon[src]) / 2) ** 2, 0.0, 1.0)))

    chosen: list[int] = []
    for b in range(2):
        lo, hi = edges[b], edges[b + 1]
        in_bin = np.where((ranges >= lo) & (ranges < hi))[0]
        target = targets[b]
        if len(in_bin) <= target:
            print(f"  Warning: band bin [{lo:.0f}, {hi:.0f}) km has only "
                  f"{len(in_bin)} candidates; requested {target}.")
            chosen.extend(in_bin.tolist())
        else:
            in_bin = in_bin[np.argsort(ranges[in_bin])]
            step = len(in_bin) / target
            chosen.extend(in_bin[[int(i * step) for i in range(target)]])

    return [(int(src[k]), int(rcv[k]), float(ranges[k])) for k in chosen]


# ── Dataset generation ────────────────────────────────────────────────────────

def generate() -> list[dict]:
    """
    Run the full sweep and write the dataset CSV.

    Loads the environment once (bathymetry, ocean nodes, sound-speed climatology),
    then loops band → month → pair → source layer → frequency. The innermost loop
    is where each training row is produced: identical geometry is passed to both the
    analytic model and Bellhop, and the two TL values are stored side by side so the
    residual can be learned later. Returns the list of row dicts (also written to
    CSV) for `plot_dataset`.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading bathymetry and ocean nodes...")
    bathy = ocean.load_bathymetry(GEBCO_PATH, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    nodes = ocean.build_ocean_nodes(bathy)
    depth_rgi = ocean.gebco_rgi(GEBCO_PATH)   # full-res seabed for Bellhop path
    shadow_rgi = ocean.grid_rgi(bathy)        # 5 km grid for shadow check
    print(f"  {len(nodes)} ocean nodes")

    print("Loading WOA23 climatology...")
    woa = prop.load_woa(TEMP_DIR, SAL_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    print(f"  {woa.lat.size}x{woa.lon.size} grid")

    rows: list[dict] = []
    t0 = time.time()
    n_bell_ok = n_bell_total = 0

    for band, min_km, max_km, n_pairs, seed in RANGE_BANDS:
        for month in MONTHS:
            mon = _MONTH_ABBR[month - 1]
            rng = np.random.default_rng(seed + month)
            pairs = select_pairs(nodes, n_pairs, min_km, max_km, rng)
            print(f"\n[{band} {mon}] {len(pairs)} pairs "
                  f"({min(r for *_, r in pairs):.0f}-{max(r for *_, r in pairs):.0f} km)")

            for pair_i, (si, ri, range_km) in enumerate(pairs):
                src, rcv = nodes[si], nodes[ri]

                path_r_km, path_dep = ocean.sample_path_bathy(
                    src.lat, src.lon, rcv.lat, rcv.lon, BATHY_STEP_KM, depth_rgi)
                dep = path_dep.copy()
                for k in range(1, len(dep)):            # forward-fill land NaNs
                    if np.isnan(dep[k]):
                        dep[k] = dep[k - 1]
                if np.isnan(dep[0]):
                    continue                            # source over land — skip
                bathy_arr = np.column_stack([path_r_km * 1000.0, dep])
                max_depth = float(np.nanmax(dep))

                # Sound-speed profile at the source; shared by every layer/frequency
                # of this pair. group_id tags all rows from this pair so fit_xgb can
                # keep them together when splitting train/test.
                c_raw, z_raw = prop.sound_speed_profile(woa, src.lat, src.lon, month)
                ssp = prepare_ssp(z_raw, c_raw, max_depth)
                group_id = f"{band}_{mon}_{pair_i}"

                for layer in src.available_layers:
                    src_depth = src.layer_source_depth_m[layer]
                    if src_depth >= src.depth_m - prop.CLEARANCE_M:
                        continue                        # source too close to seabed
                    shadow = prop.shadow_obstructed(
                        src.lat, src.lon, src_depth, rcv.lat, rcv.lon, shadow_rgi)
                    c_mean = prop.layer_mean_speed(c_raw, z_raw, layer, src.depth_m)

                    for freq_hz in FREQS_HZ:
                        tl_anal = prop.analytic_tl_point(range_km, freq_hz, layer, shadow)
                        tl_bell = run_bellhop(ssp, bathy_arr, src_depth,
                                              RECEIVER_DEPTH_M, range_km * 1000.0, freq_hz)
                        n_bell_total += 1
                        n_bell_ok += int(not math.isnan(tl_bell))

                        rows.append({
                            "band": band, "month": month, "pair_i": pair_i,
                            "group_id": group_id,
                            "src_lat": src.lat, "src_lon": src.lon,
                            "rcv_lat": rcv.lat, "rcv_lon": rcv.lon,
                            "range_km": round(range_km, 3),
                            "src_seabed_depth_m": round(src.depth_m, 1),
                            "rcv_seabed_depth_m": round(rcv.depth_m, 1),
                            "path_min_depth_m": round(float(np.nanmin(dep)), 1),
                            "path_mean_depth_m": round(float(np.nanmean(dep)), 1),
                            "layer": layer, "src_depth_m": src_depth,
                            "rcv_depth_m": RECEIVER_DEPTH_M, "freq_hz": freq_hz,
                            "layer_mean_speed_ms": round(c_mean, 2) if not math.isnan(c_mean) else float("nan"),
                            "shadow": shadow,
                            "shadow_penalty_db": prop.SHADOW_PENALTY_DB if shadow else 0.0,
                            "tl_analytic_db": round(tl_anal, 3),
                            "tl_bellhop_db": tl_bell,
                        })

            print(f"  running total: {len(rows)} rows, "
                  f"Bellhop OK {n_bell_ok}/{n_bell_total}  ({time.time()-t0:.0f}s)")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {CSV_PATH}  "
          f"(Bellhop success {100*n_bell_ok/max(n_bell_total,1):.0f}%)")
    return rows


# ── Sanity figures ────────────────────────────────────────────────────────────

def plot_dataset(rows: list[dict]) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    ranges = np.array([r["range_km"] for r in rows])
    anal = np.array([r["tl_analytic_db"] for r in rows], dtype=float)
    bell = np.array([r["tl_bellhop_db"] for r in rows], dtype=float)
    ok = np.isfinite(bell) & (bell <= 160.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for band, color in [("short", "#2166ac"), ("mid", "#4dac26"), ("long", "#d6604d")]:
        vals = ranges[[r["band"] == band for r in rows]]
        ax.hist(vals, bins=40, alpha=0.6, color=color, label=band)
    ax.set_xlabel("Range (km)"); ax.set_ylabel("Rows")
    ax.set_title("Range diversity across bands"); ax.legend()
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dataset_range_hist.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'dataset_range_hist.png')}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(anal[ok], bell[ok], s=6, alpha=0.3, color="#377eb8")
    lo = min(anal[ok].min(), bell[ok].min()); hi = max(anal[ok].max(), bell[ok].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="1:1")
    ax.set_xlabel("Analytic TL (dB)"); ax.set_ylabel("Bellhop TL (dB)")
    ax.set_title("Bellhop vs analytic TL"); ax.legend()
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dataset_residual_scatter.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'dataset_residual_scatter.png')}")
    plt.close("all")


# ── Usage ─────────────────────────────────────────────────────────────────────
# conda run -n sensor_opt python generate_dataset.py
if __name__ == "__main__":
    rows = generate()
    plot_dataset(rows)

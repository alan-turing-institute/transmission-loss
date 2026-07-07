"""
bellhop.py — Bellhop ray-tracing transmission loss for a single point.

Thin wrapper around arlpy/Bellhop that computes the incoherent one-way TL from a
source (at depth) to a surface receiver, given a sound-speed profile and the
along-path seabed profile. This is the reference "ground truth" the analytic model
is trained to match.

In the pipeline: `generate_dataset.py` calls `run_bellhop` for every geometry,
passing the sound-speed profile from `propagation` and the seabed profile from
`ocean`, and stores the result alongside the analytic TL for the same geometry.

Requires the Acoustic Toolbox `bellhop.exe` on PATH (default ~/at/Bellhop).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import xarray as xr

# Ensure bellhop.exe is discoverable (not inherited under `conda run`).
_BELLHOP_DIR = os.path.expanduser("~/at/Bellhop")
if os.path.isdir(_BELLHOP_DIR) and _BELLHOP_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _BELLHOP_DIR + os.pathsep + os.environ.get("PATH", "")

import arlpy.uwapm as pm  # noqa: E402  (import after PATH is set)


def prepare_ssp(
    ssp_depth_m: np.ndarray, ssp_speed_ms: np.ndarray, max_depth_m: float,
) -> np.ndarray:
    """
    Turn a raw WOA profile into the (N, 2) [depth_m, speed_ms] array arlpy wants.

    Covers 0..max_depth_m, drops NaNs, extends past WOA coverage by repeating the
    deepest valid speed, and guarantees >= 4 monotonic points (an arlpy
    requirement).
    """
    valid = ~np.isnan(ssp_speed_ms) & (ssp_depth_m <= max_depth_m + 1.0)
    if not valid.any():
        return np.column_stack([
            np.linspace(0.0, max_depth_m, 4),
            np.array([1480.0, 1478.0, 1476.0, 1474.0]),
        ])
    z_v, c_v = ssp_depth_m[valid].astype(float), ssp_speed_ms[valid].astype(float)
    if z_v[0] > 0.5:
        z_v = np.concatenate([[0.0], z_v]); c_v = np.concatenate([[c_v[0]], c_v])
    if z_v[-1] < max_depth_m - 1.0:
        z_v = np.append(z_v, max_depth_m); c_v = np.append(c_v, c_v[-1])
    if len(z_v) < 4:
        z_dense = np.arange(0.0, max_depth_m + 1.0, max(50.0, max_depth_m / 20.0))
        c_v = np.interp(z_dense, z_v, c_v); z_v = z_dense
    return np.column_stack([z_v, c_v])


def run_bellhop(
    ssp: np.ndarray,
    bathy: np.ndarray,
    tx_depth_m: float,
    rx_depth_m: float,
    rx_range_m: float,
    freq_hz: float,
) -> float:
    """
    Incoherent Bellhop TL (dB) for a single source→receiver point; NaN on failure.

    ssp        : (N, 2) [depth_m, speed_ms]
    bathy      : (M, 2) [range_m, depth_m] along-path seabed profile
    tx_depth_m : source depth (m)
    rx_depth_m : receiver depth (m)
    rx_range_m : receiver range from source (m)
    freq_hz    : frequency (Hz)

    arlpy returns complex pressure p; TL = -20 log10(|p|).
    """
    bathy = np.array(bathy)  # copy — don't mutate caller
    if bathy[-1, 0] < rx_range_m:                       # last range must cover rx
        bathy = np.vstack([bathy, [rx_range_m + 10.0, bathy[-1, 1]]])
    max_depth = float(np.nanmax(bathy[:, 1]))
    ssp_use = ssp.copy()
    if ssp_use[-1, 0] < max_depth:                      # ssp must reach the seabed
        ssp_use = np.vstack([ssp_use, [max_depth + 1.0, ssp_use[-1, 1]]])

    env = pm.create_env2d(
        depth=bathy, soundspeed=ssp_use,
        tx_depth=tx_depth_m, rx_depth=rx_depth_m, rx_range=float(rx_range_m),
        frequency=freq_hz, nbeams=0, min_angle=-80, max_angle=80,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pm.compute_transmission_loss(env, mode=pm.incoherent)
        if result is None:
            return float("nan")
        p = np.asarray(result, dtype=complex)
        tl = -20.0 * np.log10(np.maximum(np.abs(p), 1e-30))
        return float(tl.ravel()[0])
    except Exception as exc:
        print(f"    Bellhop error: {exc}")
        return float("nan")


# ── Test / usage example ──────────────────────────────────────────────────────
# Inputs : a flat 2000 m bottom, downward-refracting SSP, source 400 m, 100 km.
# Outputs: incoherent Bellhop TL at 300 Hz (expect ~90-110 dB).
if __name__ == "__main__":
    ssp = prepare_ssp(
        np.array([0.0, 200.0, 600.0, 1500.0]),
        np.array([1490.0, 1485.0, 1480.0, 1478.0]),
        max_depth_m=2000.0,
    )
    bathy = np.array([[0.0, 2000.0], [100_000.0, 2000.0]])
    tl = run_bellhop(ssp, bathy, tx_depth_m=400.0, rx_depth_m=10.0,
                     rx_range_m=100_000.0, freq_hz=300.0)
    print(f"Bellhop TL (300 Hz, 100 km, flat 2000 m): {tl:.1f} dB")

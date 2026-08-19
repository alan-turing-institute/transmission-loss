
import copy
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, interp1d

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os
import pandas as pd




## using the exact smae xgbOOST AS PREVIOUSLY, I want to replace is_shadow with the 8 autoencoder features and run again, then compare results


"""
xgboost_with_encoder.py — Fit an XGBoost model for the Bellhop-minus-analytic TL
residual, using the autoencoder's 8 latent bathymetry features in place of
is_shadow.

Target
------
    residual_db = tl_bellhop_db - tl_analytic_db

Split
-----
Grouped 80/20 train/test on `group_id`, same as fit_xgb.py, so the comparison
between the two models is apples-to-apples.

Inputs  : Data/BellhopData/bellhop_monthly_original.csv
          results/path_embeddings.csv   (produced by data_loading_with_existing_datapoints.ipynb)
Outputs : Data/BellhopData/tl_residual_xgb_model_encoder.joblib
          Data/BellhopData/tl_residual_xgb_importance_encoder.csv
          Figures/xgb_actual_vs_predicted_encoder.png
          Figures/xgb_feature_importance_encoder.png
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DATA_PATH = os.path.join(HERE, "Data", "BellhopData", "bellhop_monthly_original.csv")
EMBEDDINGS_PATH = os.path.join(HERE, "results", "path_embeddings.csv")
MODEL_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_model_encoder.joblib")
IMPORTANCE_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_importance_encoder.csv")
FIG_DIR = os.path.join(HERE, "Figures")

# ── Features (XGBoost handles NaN natively; only `layer` needs encoding) ───────
NUMERIC = [
    "range_km", "log10_freq_hz",
    "src_seabed_depth_m", "rcv_seabed_depth_m",
    "path_min_depth_m", "path_mean_depth_m",
    "src_depth_m", "rcv_depth_m", "layer_mean_speed_ms",
]
BINARY = ["is_shadow", "month_sin", "month_cos"]
CATEGORICAL = ["layer"]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_lambda=5.0, reg_alpha=1.0, random_state=42, n_jobs=-1, verbosity=0,
)


def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """Load the dataset, drop Bellhop artifacts, add target + engineered
    features, and merge in the autoencoder embeddings by group_id."""
    df = pd.read_csv(DATA_PATH)
    valid = (df["tl_bellhop_db"].notna() & np.isfinite(df["tl_bellhop_db"])
             & (df["tl_bellhop_db"] <= 160.0))
    print(f"Loaded {len(df)} rows; dropped {(~valid).sum()} invalid Bellhop values "
          f"-> {valid.sum()} remain")
    df = df[valid].copy()

    df["residual_db"] = df["tl_bellhop_db"] - df["tl_analytic_db"]
    df["log10_freq_hz"] = np.log10(df["freq_hz"])
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_shadow"] = (df["shadow_penalty_db"] > 0).astype(float)

    if "group_id" not in df.columns:
        df["group_id"] = df["source_file"].astype(str) + "_" + df["pair_i"].astype(str)

    #read the embeddings
    embeddings = pd.read_csv(EMBEDDINGS_PATH)
    embed_cols = [c for c in embeddings.columns if c != "group_id"]  #get coloumn names from the file


    #attach each rows own embedding by merging on group id
    before = len(df)
    df = df.merge(embeddings, on="group_id", how="inner")
    dropped = before - len(df)
    if dropped:
        print(f"{dropped} rows had no matching path embedding "
              f"(land/out-of-bounds path) -> dropped")

    return df, embed_cols


def make_model(embed_cols: list[str]) -> Pipeline:
    """XGBoost regressor with ordinal-encoded `layer`; other features passthrough."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("bin", "passthrough", BINARY),
        ("embed", "passthrough", embed_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("prep", pre), ("xgb", XGBRegressor(**XGB_PARAMS))])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    print(f"  {label:6s} R2={r2_score(y_true, y_pred):.4f}  "
          f"RMSE={mean_squared_error(y_true, y_pred) ** 0.5:.3f} dB  "
          f"MAE={mean_absolute_error(y_true, y_pred):.3f} dB")


def run() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df, embed_cols = load_and_prepare()
    FEATURES = NUMERIC + BINARY + embed_cols + CATEGORICAL  #now includes the 8 from encoder,
    X, y, groups = df[FEATURES], df["residual_db"].values, df["group_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(X_train)} rows, {groups.iloc[train_idx].nunique()} pairs")
    print(f"Test:  {len(X_test)} rows, {groups.iloc[test_idx].nunique()} pairs")

    print("\nFitting XGBoost with encoder values...")
    model = make_model(embed_cols)
    model.fit(X_train, y_train)
    print("Performance:")
    _metrics(y_train, model.predict(X_train), "train")
    _metrics(y_test, model.predict(X_test), "test")

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.named_steps["xgb"].feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(imp.to_string(index=False))
    os.makedirs(os.path.dirname(IMPORTANCE_PATH), exist_ok=True)
    imp.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Saved {IMPORTANCE_PATH}")

    # Refit on all rows and save (CV/test above is the honest estimate)
    final = make_model(embed_cols)
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    _plot(df.iloc[test_idx].copy(), y_test, model.predict(X_test), imp)


def _plot(df_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray,
          imp: pd.DataFrame) -> None:
    colors = {"surface": "#2166ac", "mid": "#4dac26", "deep": "#d6604d"}
    df_test["_pred"] = y_pred

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for layer, grp in df_test.groupby("layer"):
        ax.scatter(grp["residual_db"], grp["_pred"], s=10, alpha=0.5,
                   color=colors.get(layer, "gray"), label=layer)
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual residual (dB)"); ax.set_ylabel("Predicted residual (dB)")
    ax.set_title("XGBoost residual (encoder features) — test set"); ax.legend(markerscale=2)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_actual_vs_predicted_encoder.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_actual_vs_predicted_encoder.png')}")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(imp))
    ax.barh(y_pos, imp["importance"], color="#377eb8", alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(imp["feature"], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost feature importances (encoder features)")
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_feature_importance_encoder.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_feature_importance_encoder.png')}")
    plt.show()




# ── Usage ─────────────────────────────────────────────────────────────────────
# conda run -n sensor_opt python xgboost_with_encoder.py
if __name__ == "__main__":
    run()



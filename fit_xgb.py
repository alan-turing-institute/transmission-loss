"""
fit_xgb.py — Fit an XGBoost model for the Bellhop-minus-analytic TL residual.

Target
------
    residual_db = tl_bellhop_db - tl_analytic_db
Add the predicted residual to the analytic TL to get a Bellhop-corrected TL.

Split
-----
Grouped 80/20 train/test on `group_id` (band + month + pair), so all rows of one
geographic pair stay on the same side and the model cannot memorise pair geometry.

Inputs  : Data/BellhopData/bellhop_analytic.csv  (from generate_dataset.py)
Outputs : Data/BellhopData/tl_residual_xgb_model.joblib
          Data/BellhopData/tl_residual_xgb_importance.csv
          Figures/xgb_actual_vs_predicted.png
          Figures/xgb_feature_importance.png
"""
from __future__ import annotations

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
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "Data", "BellhopData", "bellhop_analytic.csv")
MODEL_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_model.joblib")
IMPORTANCE_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_importance.csv")
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
FEATURES = NUMERIC + BINARY + CATEGORICAL

XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_lambda=5.0, reg_alpha=1.0, random_state=42, n_jobs=-1, verbosity=0,
)


def load_and_prepare() -> pd.DataFrame:
    """Load the dataset, drop Bellhop artifacts, add target + engineered features."""
    df = pd.read_csv(DATA_PATH)
    valid = (df["tl_bellhop_db"].notna() & np.isfinite(df["tl_bellhop_db"])
             & (df["tl_bellhop_db"] <= 160.0))
    print(f"Loaded {len(df)} rows; dropped {(~valid).sum()} invalid Bellhop values "
          f"-> {valid.sum()} remain")
    df = df[valid].copy()

    df["residual_db"] = df["tl_bellhop_db"] - df["tl_analytic_db"]
    df["log10_freq_hz"] = np.log10(df["freq_hz"])
    df["is_shadow"] = (df["shadow_penalty_db"] > 0).astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "group_id" not in df.columns:
        # Legacy files (pre-`group_id`) tag geometry with `pair_i`, which repeats
        # across source_file — combine the two so pairs from different bands/months
        # aren't merged into one group.
        df["group_id"] = df["source_file"].astype(str) + "_" + df["pair_i"].astype(str)
    return df


def make_model() -> Pipeline:
    """XGBoost regressor with ordinal-encoded `layer`; other features passthrough."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("bin", "passthrough", BINARY),
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
    df = load_and_prepare()
    X, y, groups = df[FEATURES], df["residual_db"].values, df["group_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(X_train)} rows, {groups.iloc[train_idx].nunique()} pairs")
    print(f"Test:  {len(X_test)} rows, {groups.iloc[test_idx].nunique()} pairs")

    print("\nFitting XGBoost...")
    model = make_model()
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
    final = make_model()
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
    ax.set_title("XGBoost residual — test set"); ax.legend(markerscale=2)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_actual_vs_predicted.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_actual_vs_predicted.png')}")

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(imp))
    ax.barh(y_pos, imp["importance"], color="#377eb8", alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(imp["feature"], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost feature importances")
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_feature_importance.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_feature_importance.png')}")
    plt.close("all")


# ── Usage ─────────────────────────────────────────────────────────────────────
# conda run -n sensor_opt python fit_xgb.py
if __name__ == "__main__":
    run()

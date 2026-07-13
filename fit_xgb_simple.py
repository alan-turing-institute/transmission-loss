"""
fit_xgb_simple.py — Fit an XGBoost model for the Bellhop-minus-analytic TL residual.

Pedagogical version: no sklearn Pipeline / ColumnTransformer / OrdinalEncoder /
GroupShuffleSplit / metrics. Every preprocessing step is written out explicitly
with plain numpy and for loops so nothing happens inside a black-box object.
XGBoost is kept as-is (reimplementing gradient boosting is out of scope).

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
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "Data", "BellhopData", "bellhop_monthly_original.csv")
MODEL_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_model_simple.joblib")
IMPORTANCE_PATH = os.path.join(HERE, "Data", "BellhopData", "tl_residual_xgb_importance_simple.csv")
FIG_DIR = os.path.join(HERE, "Figures")

# ── Features (`layer` is the only one needing encoding, done by hand below) ───
NUMERIC = [
    "range_km", "log10_freq_hz",
    "src_seabed_depth_m", "rcv_seabed_depth_m",
    "path_min_depth_m", "path_mean_depth_m",
    "src_depth_m", "rcv_depth_m", "layer_mean_speed_ms",
]
BINARY = ["is_shadow", "month_sin", "month_cos"]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_lambda=5.0, reg_alpha=1.0, random_state=42, n_jobs=-1, verbosity=0,
)


# ── Step 1: load + engineer features ────────────────────────────────────────
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


# ── Step 2: encode `layer` by hand (ordinal, no OrdinalEncoder) ────────────
def encode_layer(layer_values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Map each distinct string in `layer` to an integer code, e.g.
    {'deep': 0, 'mid': 1, 'surface': 2}. Returns the code array and the
    sorted list of categories (categories[code] recovers the original string).
    """
    categories = sorted(layer_values.unique())
    code_of = {name: i for i, name in enumerate(categories)}

    codes = np.empty(len(layer_values), dtype=float)
    for i, value in enumerate(layer_values):
        codes[i] = code_of[value]
    return codes, categories


# ── Step 3: build the feature matrix by hand (no ColumnTransformer) ────────
def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for col in NUMERIC + BINARY:
        columns.append(df[col].to_numpy(dtype=float))
        names.append(col)

    layer_codes, categories = encode_layer(df["layer"])
    columns.append(layer_codes)
    names.append("layer")
    print(f"Encoded 'layer' -> {[(c, i) for i, c in enumerate(categories)]}")

    X = np.column_stack(columns)
    return X, names


# ── Step 4: grouped train/test split by hand (no GroupShuffleSplit) ────────
def group_train_test_split(groups: np.ndarray, test_frac: float = 0.2,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle the unique group ids, then put every row of a chosen group
    entirely in train or entirely in test — no group straddles the split."""
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_test_groups = int(round(len(shuffled) * test_frac))
    test_group_set = set(shuffled[:n_test_groups])

    is_test = np.array([g in test_group_set for g in groups])
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    return train_idx, test_idx


# ── Step 5: metrics by hand (no sklearn.metrics) ────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    print(f"  {label:6s} R2={r2_score(y_true, y_pred):.4f}  "
          f"RMSE={rmse(y_true, y_pred):.3f} dB  MAE={mae(y_true, y_pred):.3f} dB")


# ── Main ─────────────────────────────────────────────────────────────────────
def run() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_and_prepare()

    X, feature_names = build_feature_matrix(df)
    y = df["residual_db"].to_numpy(dtype=float)
    groups = df["group_id"].to_numpy()

    train_idx, test_idx = group_train_test_split(groups, test_frac=0.2, seed=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)} rows, {len(np.unique(groups[train_idx]))} pairs")
    print(f"Test:  {len(test_idx)} rows, {len(np.unique(groups[test_idx]))} pairs")

    print("\nFitting XGBoost...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print("Performance:")
    print_metrics(y_train, model.predict(X_train), "train")
    print_metrics(y_test, model.predict(X_test), "test")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(imp.to_string(index=False))
    os.makedirs(os.path.dirname(IMPORTANCE_PATH), exist_ok=True)
    imp.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Saved {IMPORTANCE_PATH}")

    # Refit on all rows and save (test metrics above are the honest estimate)
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
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
    fig.savefig(os.path.join(FIG_DIR, "xgb_actual_vs_predicted_simple.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_actual_vs_predicted_simple.png')}")

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(imp))
    ax.barh(y_pos, imp["importance"], color="#377eb8", alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(imp["feature"], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost feature importances")
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "xgb_feature_importance_simple.png"), dpi=150)
    print(f"Saved {os.path.join(FIG_DIR, 'xgb_feature_importance_simple.png')}")
    plt.close("all")


# ── Usage ─────────────────────────────────────────────────────────────────────
# conda run -n sensor_opt python fit_xgb_simple.py
if __name__ == "__main__":
    run()

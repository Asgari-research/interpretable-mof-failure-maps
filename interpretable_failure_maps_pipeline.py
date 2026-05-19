#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpretable failure-map pipeline for MOF adsorption models.

This script reproduces the tabular machine-learning workflow used to build
failure maps, local trust categories, disagreement diagnostics, manuscript
figures, supplementary figures, and source-data tables for the associated MOF
adsorption reliability study.

The implementation is intentionally self-contained: input CSV files are read
from the same directory as this script, intermediate artefacts are checkpointed,
and final outputs are written under ``failure_maps_outputs/``. The pipeline does
not parse CIF files; all analyses are based on tabular geometric, adsorption,
cluster, and topology descriptors.

Required input files, expected outputs, and reproducibility notes are documented
in the repository README and the files under ``docs/``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
import time
import textwrap
import traceback
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME = "failure_maps"
OUTPUT_ROOT_NAME = "failure_maps_outputs"

# Files expected in the same folder as the script
MASTER_FILE = "clean_data.csv"
GEO_CLUSTER_FILE = "geo-clusters.csv"
MC_CLUSTER_FILE = "mc-clusters.csv"
FUNC_CLUSTER_FILE = "func-clusters.csv"
FLIG_CLUSTER_FILE = "flig-clusters.csv"
TOPOLOGY_FILE = "all_topology_lists.csv"

# Default targets
TARGET_COLUMNS = [
    "uptake(mmol/g) CO2 at 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar",
    "uptake(mmol/g) methane at 65 bar",
]

# Manuscript defaults
PRIMARY_TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
SI_SECONDARY_TARGET = "uptake(mmol/g) methane at 65 bar"

# Split settings
N_OUTER_SPLITS = 5
TEST_SIZE = 0.20
BASE_RANDOM_SEED = 42

# Processor control: --n_jobs default is 0 ("auto-safe": CPUs minus one).
# Use --n_jobs 1 for single-core, --n_jobs -1 for all CPUs, or --n_jobs N for exact workers.
N_JOBS = 0

# Thresholds and display settings
TOP_FRACTION = 0.10                 # "elite" top-k threshold: top 10%
MIN_GROUP_SIZE = 30                 # minimum group size for reporting group metrics
MAX_CATEGORIES_FOR_PLOTS = 20       # reduce clutter in figures
TOP_LEADERBOARD_N = 15              # hardest groups displayed in leaderboard
TRUST_BIN_MIN_COUNT = 20            # minimum count per 2D cell in trust atlas

# Model panel
# -----------
# The panel combines a linear baseline with two nonlinear tabular regressors.
# Hyperparameters are fixed for reproducibility and computational tractability;
# the goal is reliability mapping rather than exhaustive model optimisation.
MODEL_SPECS = {
    # Ridge is already very fast and remains unchanged as the linear baseline.
    "ridge": Ridge(alpha=1.0, random_state=None),

    # Lighter RF: fewer trees, controlled depth, larger leaves, and row
    # subsampling. This remains a credible nonlinear tabular baseline but is
    # much faster than 300 unrestricted trees on hundreds of thousands of MOFs.
    "rf": RandomForestRegressor(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.70,
        bootstrap=True,
        max_samples=0.70,
        n_jobs=1,  # overwritten at runtime by configure_runtime()
        random_state=BASE_RANDOM_SEED,
    ),

    # Lighter HGB: fewer boosting iterations, shallower trees, and early
    # stopping. HGB is still the main efficient nonlinear baseline and usually
    # remains competitive for large tabular datasets.
    "hgb": HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_depth=6,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
        tol=1e-7,
        random_state=BASE_RANDOM_SEED,
    ),
}

# Resume / overwrite flags
FORCE_RERUN_MERGE = False
FORCE_RERUN_SPLITS = False
FORCE_RERUN_MODEL_FITS = False
FORCE_RERUN_FINAL_ASSETS = False

# Whether to create parquet in addition to CSV/PKL if pyarrow exists
SAVE_PARQUET_IF_AVAILABLE = True


# =============================================================================
# Path utilities
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / OUTPUT_ROOT_NAME

DIRS = {
    "logs": OUTPUT_ROOT / "logs",
    "checkpoints": OUTPUT_ROOT / "checkpoints",
    "data_processed": OUTPUT_ROOT / "data_processed",
    "split_definitions": OUTPUT_ROOT / "split_definitions",
    "models": OUTPUT_ROOT / "results" / "models",
    "predictions": OUTPUT_ROOT / "results" / "predictions",
    "metrics": OUTPUT_ROOT / "results" / "metrics",
    "tables": OUTPUT_ROOT / "results" / "tables",
    "tables_rendered": OUTPUT_ROOT / "results" / "tables_rendered",
    "manuscript_figures": OUTPUT_ROOT / "manuscript_assets" / "figures",
    "si_figures": OUTPUT_ROOT / "supplementary_assets" / "figures",
    "figure_data": OUTPUT_ROOT / "results" / "figure_numeric_data",
    "manuscript_tables": OUTPUT_ROOT / "manuscript_assets" / "tables",
    "si_tables": OUTPUT_ROOT / "supplementary_assets" / "tables",
    "exports": OUTPUT_ROOT / "final_exports",
}

for _name, _path in DIRS.items():
    _path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Runtime configuration helpers
# =============================================================================

def resolve_n_jobs(n_jobs_value: int) -> int:
    """
    Convert user-facing --n_jobs into a valid scikit-learn n_jobs value.
    Convention: 0 = auto-safe (logical CPUs minus one), -1 = all CPUs, >0 = exact.
    """
    if n_jobs_value == -1:
        return -1
    if n_jobs_value == 0:
        detected = multiprocessing.cpu_count() or 1
        return max(1, detected - 1)
    if n_jobs_value > 0:
        return int(n_jobs_value)
    raise ValueError("n_jobs must be -1, 0, or a positive integer.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Interpretable Failure Maps ARC-MOF CSV pipeline.")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS,
        help="Default 0 means auto-safe CPU use (CPUs minus one). Use 1, -1, or a positive integer.")
    parser.add_argument("--force_model_fits", action="store_true",
        help="Ignore completed fit markers and refit all target/model/split jobs.")
    parser.add_argument("--force_merge", action="store_true",
        help="Recreate the merged master table even if a processed copy exists.")
    parser.add_argument("--force_splits", action="store_true",
        help="Recreate split definitions even if saved split JSON files exist.")
    parser.add_argument("--skip_figures", action="store_true",
        help="Run data/model aggregation and tables but skip figure generation.")
    return parser.parse_args(argv)


def configure_runtime(args: argparse.Namespace) -> int:
    """Apply command-line runtime choices to global configuration."""
    global FORCE_RERUN_MODEL_FITS, FORCE_RERUN_MERGE, FORCE_RERUN_SPLITS
    resolved = resolve_n_jobs(int(args.n_jobs))
    MODEL_SPECS["rf"].set_params(n_jobs=resolved)
    FORCE_RERUN_MODEL_FITS = bool(args.force_model_fits)
    FORCE_RERUN_MERGE = bool(args.force_merge)
    FORCE_RERUN_SPLITS = bool(args.force_splits)
    return resolved


# =============================================================================
# Logging and checkpoint helpers
# =============================================================================

LOG_FILE = DIRS["logs"] / "run_log.txt"


def log(message: str) -> None:
    """Print and append a timestamped message to the run log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_status_message(message: str) -> None:
    """Log a warning if the main logger exists; otherwise print.

    This helper is deliberately independent from the normal table-saving
    routines so it can be used inside low-level save functions without creating
    recursive save/log failures.
    """
    try:
        logger = globals().get("log", None)
        if callable(logger):
            logger(message)
        else:
            print(message, flush=True)
    except Exception:
        try:
            print(message, flush=True)
        except Exception:
            pass


def _unique_tmp_path(path: Path) -> Path:
    """Create a unique temporary path next to the destination file.

    A fixed name such as ``file.csv.tmp`` can itself become locked on Windows
    after a failed run, by antivirus indexing, or by a preview pane. Using a
    unique temporary path avoids collisions with stale ``.tmp`` files.
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _replace_with_retries(tmp_path: Path, path: Path, *, n_retries: int = 20, sleep_seconds: float = 0.75) -> Path:
    """Atomically replace ``path`` with ``tmp_path`` when possible.

    Replace ``path`` with ``tmp_path`` while tolerating transient file locks.

    If the destination cannot be replaced after repeated attempts, the temporary
    file is saved as a timestamped fallback. This preserves completed model or
    aggregation outputs even when an external process has locked the original
    file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, n_retries + 1):
        try:
            os.replace(tmp_path, path)
            return path
        except PermissionError as exc:
            last_error = exc
            # Allow transient file locks to clear before retrying.
            if attempt < n_retries:
                time.sleep(sleep_seconds)
                continue
        except OSError as exc:
            last_error = exc
            if attempt < n_retries:
                time.sleep(sleep_seconds)
                continue

    # Preserve the completed output by writing a versioned fallback file.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fallback = path.with_name(f"{path.stem}__LOCKED_DESTINATION_{timestamp}_{uuid.uuid4().hex[:8]}{path.suffix}")
    try:
        os.replace(tmp_path, fallback)
        _safe_status_message(
            "WARNING: Could not replace locked output file after retries: "
            f"{path}. A new version was saved instead: {fallback}. "
            "Close the locked file in Excel/Preview/OneDrive and, if desired, "
            "rename the fallback file to the original name."
        )
        return fallback
    except Exception as fallback_error:
        # Preserve the original reason if possible, but do not hide the fallback
        # failure because it means the data were not saved.
        raise RuntimeError(
            f"Could not save output file '{path}'. Last replace error: {last_error}. "
            f"Fallback save also failed: {fallback_error}."
        ) from fallback_error


def safe_joblib_dump(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(path)
    try:
        joblib.dump(obj, tmp_path)
        return _replace_with_retries(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Save a CSV with retry and fallback handling for locked destinations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(path)
    try:
        df.to_csv(tmp_path, index=index)
        return _replace_with_retries(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def safe_to_pickle(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(path)
    try:
        pd.to_pickle(obj, tmp_path)
        return _replace_with_retries(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def maybe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    if not SAVE_PARQUET_IF_AVAILABLE:
        return
    try:
        import pyarrow  # noqa: F401
        tmp_path = _unique_tmp_path(path)
        try:
            df.to_parquet(tmp_path, index=False)
            _replace_with_retries(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
    except Exception as exc:
        _safe_status_message(f"Note: parquet save skipped for {path.name}: {exc}")

def completed_marker_path(target_slug: str, model_name: str, split_id: int) -> Path:
    return DIRS["checkpoints"] / "fits" / f"{target_slug}__{model_name}__split{split_id:02d}.json"


def save_completed_marker(target_slug: str, model_name: str, split_id: int, payload: dict) -> None:
    save_json(payload, completed_marker_path(target_slug, model_name, split_id))


def has_completed_marker(target_slug: str, model_name: str, split_id: int) -> bool:
    return completed_marker_path(target_slug, model_name, split_id).exists()


# =============================================================================
# General utilities
# =============================================================================

def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_")
    return text.lower()


def normalize_filename(value: object) -> str:
    """
    Normalize structure identifiers to maximize merge success across tables.

    Strategy:
    * keep only basename,
    * remove .cif suffix if present,
    * strip whitespace.
    """
    s = "" if pd.isna(value) else str(value)
    s = os.path.basename(s).strip()
    s = re.sub(r"\.cif", "", s, flags=re.IGNORECASE)
    return s


def infer_cluster_column(df: pd.DataFrame) -> Optional[str]:
    """
    Infer the cluster/group column from a cluster CSV.
    Usually expected: cluster_id
    """
    preferred = [c for c in ["cluster_id", "cluster", "group", "label"] if c in df.columns]
    if preferred:
        return preferred[0]
    candidates = [c for c in df.columns if c.lower() not in {"filename", "database", "name"}]
    return candidates[0] if candidates else None


def nearest_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def rank_pct(series: pd.Series) -> pd.Series:
    """Percentile rank scaled to [0, 1]."""
    return series.rank(pct=True, method="average")


def spearman_safe(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho) if np.isfinite(rho) else np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_list(value) -> List:
    if isinstance(value, list):
        return value
    return [value]


def describe_group_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def normalize_category_value(value: object) -> str:
    """Return a safe uniform string representation for categorical ML columns."""
    if pd.isna(value):
        return "__missing__"
    text_value = str(value).strip()
    if text_value == "" or text_value.lower() in {"nan", "none", "nat", "<na>"}:
        return "__missing__"
    try:
        numeric_value = float(text_value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    except Exception:
        pass
    return text_value


def sanitize_categorical_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert selected columns to clean string categories in a copied DataFrame."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(normalize_category_value).astype("object")
    return out


def clean_design_matrix(X: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    """Coerce numeric features to float and categorical features to safe strings."""
    out = X.copy()
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].map(normalize_category_value).astype("object")
    return out


def save_column_diagnostics(df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> None:
    """Save a compact diagnostic table for model feature columns."""
    rows = []
    for col in list(numeric_cols) + list(categorical_cols):
        if col not in df.columns:
            continue
        ser = df[col]
        rows.append({
            "column": col,
            "role": "categorical" if col in categorical_cols else "numeric",
            "dtype": str(ser.dtype),
            "n_missing": int(ser.isna().sum()),
            "n_unique_including_missing": int(ser.astype("object").fillna("__missing__").nunique()),
            "example_values": "; ".join(map(str, ser.dropna().astype("object").head(5).tolist())),
        })
    if rows:
        safe_to_csv(pd.DataFrame(rows), DIRS["tables"] / "feature_column_diagnostics.csv", index=False)


# =============================================================================
# Project-specific feature engineering
# =============================================================================

BASE_NUMERIC_FEATURES = [
    "Density",
    "UC_volume",
    "ASA",
    "vASA",
    "gASA",
    "NASA",
    "gNASA",
    "vNASA",
    "AVA",
    "AVAf",
    "AVAg",
    "NAVA",
    "NAVAf",
    "NAVAg",
    "POAVA",
    "POAVAf",
    "POAVAg",
    "NPOAVA",
    "NPOAVAf",
    "NPOAVAg",
    "Df",
    "Di",
    "Dif",
]

ENGINEERED_NUMERIC_FEATURES = [
    "lcd_pld_ratio",
    "cavity_window_gap",
    "sa_pv_ratio",
    "vf_density_ratio",
    "log_pld_plus1",
    "log_lcd_plus1",
    "avaf_x_density",
    "pore_shape_ratio",
]

CATEGORICAL_FEATURES = [
    "topology_group",
    "topology_frequency_group",
    "pore_regime",
    "density_regime",
    "geometry_family",
    "metal_family",
    "functional_family",
    "linker_family",
]

NOVELTY_FEATURES = [
    "Density",
    "Df",
    "Di",
    "Dif",
    "ASA",
    "AVA",
    "AVAf",
    "POAVA",
    "POAVAf",
]

TRUST_MAP_AXES = ("Df", "Density")


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eps = 1e-9
    if {"Di", "Df"}.issubset(out.columns):
        out["lcd_pld_ratio"] = out["Di"] / (out["Df"] + eps)
        out["cavity_window_gap"] = out["Di"] - out["Df"]
        out["log_pld_plus1"] = np.log1p(np.clip(out["Df"], a_min=0, a_max=None))
        out["log_lcd_plus1"] = np.log1p(np.clip(out["Di"], a_min=0, a_max=None))
    else:
        out["lcd_pld_ratio"] = np.nan
        out["cavity_window_gap"] = np.nan
        out["log_pld_plus1"] = np.nan
        out["log_lcd_plus1"] = np.nan

    if {"ASA", "AVA"}.issubset(out.columns):
        out["sa_pv_ratio"] = out["ASA"] / (out["AVA"] + eps)
    else:
        out["sa_pv_ratio"] = np.nan

    if {"AVAf", "Density"}.issubset(out.columns):
        out["vf_density_ratio"] = out["AVAf"] / (out["Density"] + eps)
        out["avaf_x_density"] = out["AVAf"] * out["Density"]
    else:
        out["vf_density_ratio"] = np.nan
        out["avaf_x_density"] = np.nan

    if {"Di", "Df", "Dif"}.issubset(out.columns):
        out["pore_shape_ratio"] = out["Dif"] / (out["Di"] + eps)
    else:
        out["pore_shape_ratio"] = np.nan

    return out


def assign_pore_regime(df: pd.DataFrame) -> pd.Series:
    """
    Simple interpretable pore-size regime based on PLD (Df) and LCD (Di).
    These thresholds are heuristic and designed for transparent, reproducible
    binning rather than claiming a universal physical taxonomy.
    """
    dfi = df["Df"]
    dii = df["Di"]

    conditions = [
        (dfi < 3.5),
        (dfi >= 3.5) & (dfi < 6.0),
        (dfi >= 6.0) & (dfi < 10.0),
        (dfi >= 10.0),
    ]
    labels = [
        "ultramicroporous_or_tight",
        "small_pore",
        "medium_pore",
        "large_pore",
    ]
    regime = np.select(conditions, labels, default="unclassified")

    # augment with cavity contrast for interpretability
    if "Dif" in df.columns:
        diff = df["Dif"].fillna(0.0)
        regime = pd.Series(regime, index=df.index).astype(str)
        regime = np.where(diff >= 4.0, pd.Series(regime).astype(str) + "__high_constriction_contrast", regime)

    return pd.Series(regime, index=df.index, name="pore_regime")


def assign_density_regime(df: pd.DataFrame) -> pd.Series:
    density = df["Density"]
    bins = [-np.inf, 0.40, 0.70, 1.00, np.inf]
    labels = ["very_low_density", "low_density", "moderate_density", "high_density"]
    return pd.cut(density, bins=bins, labels=labels).astype("object").fillna("missing")


def collapse_rare_categories(series: pd.Series, min_count: int = 100, other_label: str = "other_rare") -> pd.Series:
    # Normalize to strings first to avoid pandas/scikit-learn mixed-type warnings.
    out = series.map(normalize_category_value).astype("object")
    counts = out.value_counts(dropna=False)
    keep = set(counts[counts >= min_count].index.tolist())
    out = out.where(out.isin(keep), other_label).astype("object")
    return out


def compute_topology_frequency_group(df: pd.DataFrame, topology_col: str = "Crystalnet") -> pd.Series:
    top = df[topology_col].astype("object").fillna("missing")
    counts = top.value_counts()
    out = pd.Series(index=df.index, dtype="object")
    out.loc[top.map(counts) >= 1000] = "very_common_topology"
    out.loc[(top.map(counts) >= 250) & (top.map(counts) < 1000)] = "common_topology"
    out.loc[(top.map(counts) >= 50) & (top.map(counts) < 250)] = "uncommon_topology"
    out.loc[top.map(counts) < 50] = "rare_topology"
    out = out.fillna("missing")
    return out


def prepare_model_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = nearest_existing_columns(df, BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES)
    categorical_cols = nearest_existing_columns(df, CATEGORICAL_FEATURES)
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=1,  # keep preprocessing sequential and predictable on Windows/Python 3.13
    )
    return transformer


def build_model_pipeline(model_name: str, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    model = clone(MODEL_SPECS[model_name])
    pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ]
    )
    return pipe


# =============================================================================
# Data loading and merge
# =============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    # low_memory=False avoids pandas DtypeWarning on mixed-type ARC/cluster columns.
    return pd.read_csv(path, low_memory=False)


def load_and_merge_data(force_rerun: bool = False) -> pd.DataFrame:
    merged_csv = DIRS["data_processed"] / "merged_master_table.csv"
    merged_pkl = DIRS["data_processed"] / "merged_master_table.pkl"

    if merged_pkl.exists() and not force_rerun:
        log("Loading previously saved merged master table.")
        cached = pd.read_pickle(merged_pkl)
        cached = sanitize_categorical_columns(cached, CATEGORICAL_FEATURES + [
            "Crystalnet", "likely_topology", "geo_cluster_id", "mc_cluster_id",
            "func_cluster_id", "flig_cluster_id",
        ])
        return cached

    log("Loading input CSV files.")
    master = load_csv(SCRIPT_DIR / MASTER_FILE)
    geo = load_csv(SCRIPT_DIR / GEO_CLUSTER_FILE)
    mc = load_csv(SCRIPT_DIR / MC_CLUSTER_FILE)
    func = load_csv(SCRIPT_DIR / FUNC_CLUSTER_FILE)
    flig = load_csv(SCRIPT_DIR / FLIG_CLUSTER_FILE)
    topo = load_csv(SCRIPT_DIR / TOPOLOGY_FILE)

    log(f"Master rows: {len(master):,}")

    # Normalize filenames
    master = master.copy()
    master["id_norm"] = master["filename"].map(normalize_filename)

    for df, label in [(geo, "geo"), (mc, "mc"), (func, "func"), (flig, "flig")]:
        df["id_norm"] = df["filename"].map(normalize_filename)
        cluster_col = infer_cluster_column(df)
        if cluster_col is None:
            raise ValueError(f"Could not infer cluster column for {label} cluster file.")
        renamed = df[["id_norm", cluster_col]].drop_duplicates("id_norm").rename(
            columns={cluster_col: f"{label}_cluster_id"}
        )
        master = master.merge(renamed, on="id_norm", how="left")
        log(f"Merged {label} clusters. Coverage = {master[f'{label}_cluster_id'].notna().mean():.3%}")

    # Optional topology supplement
    topo = topo.copy()
    if "Name" in topo.columns:
        topo["id_norm_from_name"] = topo["Name"].map(normalize_filename)
    if "filename" in topo.columns:
        topo["id_norm_from_filename"] = topo["filename"].map(normalize_filename)

    topo_id_col = None
    if "id_norm_from_name" in topo.columns:
        topo_id_col = "id_norm_from_name"
    elif "id_norm_from_filename" in topo.columns:
        topo_id_col = "id_norm_from_filename"

    if topo_id_col is not None:
        topo_keep_cols = [topo_id_col]
        for c in ["Crystalnet", "likely topology"]:
            if c in topo.columns:
                topo_keep_cols.append(c)
        topo_small = topo[topo_keep_cols].drop_duplicates(topo_id_col).rename(columns={topo_id_col: "id_norm"})
        master = master.merge(topo_small, on="id_norm", how="left", suffixes=("", "_toposupp"))

    # Consolidate topology columns
    if "Crystalnet_toposupp" in master.columns:
        master["Crystalnet"] = master["Crystalnet"].fillna(master["Crystalnet_toposupp"])
    if "likely topology" in master.columns:
        master["likely_topology"] = master["likely topology"]
    else:
        master["likely_topology"] = master.get("Crystalnet", pd.Series(index=master.index, dtype="object"))

    # Remove obvious unnamed columns from later analysis but keep original file provenance if useful
    unnamed_cols = [c for c in master.columns if c.lower().startswith("unnamed:")]
    if unnamed_cols:
        log(f"Retaining but not using unnamed columns: {unnamed_cols}")

    # Engineer features
    master = create_engineered_features(master)

    # Add interpretable group labels
    master["pore_regime"] = assign_pore_regime(master)
    master["density_regime"] = assign_density_regime(master)
    master["topology_group"] = collapse_rare_categories(master["Crystalnet"].astype("object").fillna("missing"), min_count=100)
    master["topology_frequency_group"] = compute_topology_frequency_group(master, topology_col="Crystalnet")
    master["geometry_family"] = collapse_rare_categories(master["geo_cluster_id"].astype("object"), min_count=100, other_label="other_geometry_families")
    master["metal_family"] = collapse_rare_categories(master["mc_cluster_id"].astype("object"), min_count=100, other_label="other_metal_families")
    master["functional_family"] = collapse_rare_categories(master["func_cluster_id"].astype("object"), min_count=100, other_label="other_functional_families")
    master["linker_family"] = collapse_rare_categories(master["flig_cluster_id"].astype("object"), min_count=100, other_label="other_linker_families")

    # Ensure all categorical columns are uniformly strings before they are cached
    # or passed to OneHotEncoder. This avoids mixed str/float category failures.
    master = sanitize_categorical_columns(master, CATEGORICAL_FEATURES + [
        "Crystalnet", "likely_topology", "geo_cluster_id", "mc_cluster_id",
        "func_cluster_id", "flig_cluster_id",
    ])

    # Save merged
    safe_to_csv(master, merged_csv, index=False)
    safe_to_pickle(master, merged_pkl)
    maybe_to_parquet(master, DIRS["data_processed"] / "merged_master_table.parquet")
    log("Merged master table saved.")
    return master


# =============================================================================
# Split persistence
# =============================================================================

def make_target_cohort(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    feature_cols_needed = nearest_existing_columns(df, ["Density", "Df", "Di", "ASA", "AVA", "AVAf", "POAVA"])
    keep_cols = ["id_norm", "filename", target_col] + feature_cols_needed
    cohort = df[df[target_col].notna()].copy()

    # Keep rows where essential geometry is present
    if feature_cols_needed:
        essential_mask = cohort[feature_cols_needed].notna().all(axis=1)
        cohort = cohort.loc[essential_mask].copy()
    return cohort


def build_and_save_splits(df: pd.DataFrame, target_col: str, force_rerun: bool = False) -> List[dict]:
    target_slug = slugify(target_col)
    split_path = DIRS["split_definitions"] / f"{target_slug}_outer_splits.json"

    if split_path.exists() and not force_rerun:
        log(f"Loading split definitions for target: {target_col}")
        return load_json(split_path)["splits"]

    cohort = make_target_cohort(df, target_col)
    n = len(cohort)
    if n < 100:
        raise ValueError(f"Too few samples for target '{target_col}' after cohort filtering: {n}")

    splitter = ShuffleSplit(
        n_splits=N_OUTER_SPLITS,
        test_size=TEST_SIZE,
        random_state=BASE_RANDOM_SEED,
    )

    splits = []
    idx = np.arange(n)
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(idx)):
        split_record = {
            "split_id": split_id,
            "train_positions": train_idx.tolist(),
            "test_positions": test_idx.tolist(),
            "cohort_row_count": int(n),
            "random_seed": BASE_RANDOM_SEED,
            "test_size": TEST_SIZE,
        }
        splits.append(split_record)

    save_json({"target": target_col, "splits": splits}, split_path)
    log(f"Saved split definitions for {target_col}")
    return splits



def compute_rank_based_quantities(y_train: pd.Series, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute percentile/rank style quantities for failure analysis.

    Metrics:
    * true percentile rank within train+test-like reference (training only reference)
    * predicted percentile rank by comparing y_pred to train target distribution
    * absolute percentile rank error
    * elite misclassification under top fraction threshold
    """
    train_values = np.asarray(y_train.dropna())
    if len(train_values) == 0:
        raise ValueError("Training target values are empty in rank-based calculation.")

    elite_threshold = np.quantile(train_values, 1.0 - TOP_FRACTION)

    # Compare true y_test and predicted y_pred to training distribution
    true_pct = np.array([(train_values <= val).mean() for val in y_test.to_numpy()])
    pred_pct = np.array([(train_values <= val).mean() for val in y_pred])

    true_is_elite = y_test.to_numpy() >= elite_threshold
    pred_is_elite = y_pred >= elite_threshold

    out = pd.DataFrame(
        {
            "true_percentile_rank": true_pct,
            "pred_percentile_rank": pred_pct,
            "abs_percentile_rank_error": np.abs(true_pct - pred_pct),
            "elite_threshold_train": elite_threshold,
            "true_is_elite": true_is_elite.astype(int),
            "pred_is_elite": pred_is_elite.astype(int),
            "elite_misclassified": (true_is_elite != pred_is_elite).astype(int),
            "elite_false_negative": ((true_is_elite == 1) & (pred_is_elite == 0)).astype(int),
            "elite_false_positive": ((true_is_elite == 0) & (pred_is_elite == 1)).astype(int),
        }
    )
    return out


def compute_novelty_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    novelty_cols: Sequence[str],
    rarity_reference_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Novelty proxies without CIF parsing:
    * nearest-neighbor distance in descriptor space,
    * mean distance to k nearest neighbors,
    * local training density proxy = inverse of mean kNN distance,
    * cluster rarity proxies,
    * topology rarity proxy.
    """
    novelty_cols = [c for c in novelty_cols if c in train_df.columns and c in test_df.columns]
    if len(novelty_cols) == 0:
        return pd.DataFrame(index=test_df.index)

    # Median-impute using training medians
    train_X = train_df[novelty_cols].copy()
    test_X = test_df[novelty_cols].copy()
    med = train_X.median()
    train_X = train_X.fillna(med)
    test_X = test_X.fillna(med)

    # Standardize manually using training stats
    mu = train_X.mean()
    sigma = train_X.std(ddof=0).replace(0, 1.0)
    train_Z = (train_X - mu) / sigma
    test_Z = (test_X - mu) / sigma

    k = min(10, len(train_Z))
    if k < 2:
        return pd.DataFrame(index=test_df.index)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_Z)
    distances, _ = nn.kneighbors(test_Z)

    out = pd.DataFrame(index=test_df.index)
    out["nn_distance_1"] = distances[:, 0]
    out["nn_distance_mean_k"] = distances.mean(axis=1)
    out["local_training_density"] = 1.0 / (distances.mean(axis=1) + 1e-9)

    # rarity proxies from training frequencies
    for col in rarity_reference_cols:
        if col in train_df.columns and col in test_df.columns:
            train_freq = train_df[col].astype("object").fillna("missing").value_counts(normalize=True)
            test_vals = test_df[col].astype("object").fillna("missing")
            out[f"{col}_freq_train"] = test_vals.map(train_freq).fillna(0.0)
            out[f"{col}_rarity_train"] = 1.0 - out[f"{col}_freq_train"]

    return out


def calculate_local_trust_category(abs_error: pd.Series, disagreement: pd.Series) -> pd.Series:
    """
    Define local trust categories from error/disagreement quantiles.
    Intended for interpretation, not as a calibrated uncertainty guarantee.
    """
    err_q1 = abs_error.quantile(0.33)
    err_q2 = abs_error.quantile(0.66)
    dis_q1 = disagreement.quantile(0.33)
    dis_q2 = disagreement.quantile(0.66)

    cats = []
    for e, d in zip(abs_error, disagreement):
        if pd.isna(e) or pd.isna(d):
            cats.append("missing")
        elif e <= err_q1 and d <= dis_q1:
            cats.append("easy_and_stable")
        elif e >= err_q2 and d >= dis_q2:
            cats.append("hard_and_unstable")
        elif e >= err_q2 and d < dis_q2:
            cats.append("hard_but_consistent")
        elif e < err_q2 and d >= dis_q2:
            cats.append("ambiguous_or_model_sensitive")
        else:
            cats.append("intermediate")
    return pd.Series(cats, index=abs_error.index)


# =============================================================================
# Fit-evaluate-save loop
# =============================================================================

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_test": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(spearman_safe(pd.Series(y_true), pd.Series(y_pred))),
    }


def fit_one_job(
    df: pd.DataFrame,
    target_col: str,
    split_record: dict,
    model_name: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    force_rerun: bool = False,
) -> Optional[dict]:
    """
    Fit one target-model-split job, save everything needed for full downstream analysis.
    """
    target_slug = slugify(target_col)
    split_id = split_record["split_id"]

    if has_completed_marker(target_slug, model_name, split_id) and not force_rerun:
        log(f"Skipping completed job: target={target_col} model={model_name} split={split_id}")
        return None

    cohort = make_target_cohort(df, target_col).reset_index(drop=True)
    train_idx = np.array(split_record["train_positions"], dtype=int)
    test_idx = np.array(split_record["test_positions"], dtype=int)

    train_df = cohort.iloc[train_idx].copy()
    test_df = cohort.iloc[test_idx].copy()

    X_train = clean_design_matrix(train_df[numeric_cols + categorical_cols].copy(), numeric_cols, categorical_cols)
    X_test = clean_design_matrix(test_df[numeric_cols + categorical_cols].copy(), numeric_cols, categorical_cols)
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).copy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").astype(float).copy()

    valid_train = y_train.notna()
    valid_test = y_test.notna()
    if not valid_train.all():
        X_train = X_train.loc[valid_train].copy()
        train_df = train_df.loc[valid_train].copy()
        y_train = y_train.loc[valid_train].copy()
    if not valid_test.all():
        X_test = X_test.loc[valid_test].copy()
        test_df = test_df.loc[valid_test].copy()
        y_test = y_test.loc[valid_test].copy()
    if len(y_train) < 10 or len(y_test) < 5:
        raise ValueError(f"Too few valid train/test rows after numeric target cleaning for {target_col}.")

    pipe = build_model_pipeline(model_name, numeric_cols, categorical_cols)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    y_pred = pipe.predict(X_test)

    # prediction-level outputs
    pred_keep_cols = [
        "id_norm", "filename", target_col, "Crystalnet", "pore_regime", "density_regime",
        "topology_group", "topology_frequency_group", "geometry_family", "metal_family",
        "functional_family", "linker_family", "Density", "Df", "Di", "Dif",
    ]
    pred_keep_cols = [c for c in pred_keep_cols if c in test_df.columns]
    pred_df = test_df[pred_keep_cols].copy()

    pred_df = pred_df.rename(columns={target_col: "y_true"})
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["squared_error"] = pred_df["residual"] ** 2
    pred_df["split_id"] = split_id
    pred_df["model_name"] = model_name
    pred_df["target"] = target_col

    rank_df = compute_rank_based_quantities(y_train=y_train, y_test=y_test, y_pred=y_pred)
    pred_df = pd.concat([pred_df.reset_index(drop=True), rank_df.reset_index(drop=True)], axis=1)

    novelty_df = compute_novelty_features(
        train_df=train_df,
        test_df=test_df,
        novelty_cols=NOVELTY_FEATURES,
        rarity_reference_cols=[
            "geometry_family",
            "metal_family",
            "functional_family",
            "linker_family",
            "topology_group",
            "topology_frequency_group",
        ],
    )
    pred_df = pd.concat([pred_df.reset_index(drop=True), novelty_df.reset_index(drop=True)], axis=1)

    # save model and predictions
    pred_path = DIRS["predictions"] / target_slug / f"{model_name}__split{split_id:02d}_predictions.csv"
    model_path = DIRS["models"] / target_slug / f"{model_name}__split{split_id:02d}.joblib"

    safe_to_csv(pred_df, pred_path, index=False)
    safe_joblib_dump(pipe, model_path)

    metrics = compute_basic_metrics(y_true=y_test.to_numpy(), y_pred=y_pred)
    metrics.update(
        {
            "target": target_col,
            "target_slug": target_slug,
            "model_name": model_name,
            "split_id": split_id,
            "fit_seconds": fit_seconds,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }
    )
    metrics_path = DIRS["metrics"] / target_slug / f"{model_name}__split{split_id:02d}_metrics.json"
    save_json(metrics, metrics_path)

    save_completed_marker(
        target_slug,
        model_name,
        split_id,
        payload={
            "status": "done",
            "metrics_path": str(metrics_path),
            "predictions_path": str(pred_path),
            "model_path": str(model_path),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    log(
        f"Finished job: target={target_col} model={model_name} split={split_id} "
        f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
    )
    return metrics


def run_model_panel(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols, categorical_cols = prepare_model_features(df)
    log(f"Using {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.")
    save_column_diagnostics(df, numeric_cols, categorical_cols)

    all_metrics = []

    for target_col in TARGET_COLUMNS:
        log(f"Preparing target: {target_col}")
        splits = build_and_save_splits(df, target_col, force_rerun=FORCE_RERUN_SPLITS)
        target_slug = slugify(target_col)

        for model_name in MODEL_SPECS:
            for split_record in splits:
                try:
                    result = fit_one_job(
                        df=df,
                        target_col=target_col,
                        split_record=split_record,
                        model_name=model_name,
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        force_rerun=FORCE_RERUN_MODEL_FITS,
                    )
                    if result is not None:
                        all_metrics.append(result)
                except Exception as exc:
                    err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    log(f"ERROR in target={target_col}, model={model_name}, split={split_record['split_id']}\n{err_text}")
                    continue

        # consolidate metrics for this target from all saved files
        metrics_files = sorted((DIRS["metrics"] / target_slug).glob("*_metrics.json"))
        target_metrics = [load_json(p) for p in metrics_files]
        if target_metrics:
            target_metrics_df = pd.DataFrame(target_metrics)
            safe_to_csv(target_metrics_df, DIRS["metrics"] / target_slug / "all_metrics_for_target.csv", index=False)

    # global consolidated metrics
    all_metric_files = sorted(DIRS["metrics"].glob("*/*.json"))
    all_metric_rows = [load_json(p) for p in all_metric_files if p.name.endswith("_metrics.json")]
    metrics_df = pd.DataFrame(all_metric_rows)
    if metrics_df.empty:
        raise RuntimeError(
            "No successful model fits were found. Check logs/run_log.txt. "
            "The most common cause is invalid feature preprocessing or missing input files."
        )
    metrics_df = metrics_df.sort_values(["target", "model_name", "split_id"]).reset_index(drop=True)
    safe_to_csv(metrics_df, DIRS["tables"] / "all_split_metrics.csv", index=False)
    safe_to_pickle(metrics_df, DIRS["tables"] / "all_split_metrics.pkl")
    maybe_to_parquet(metrics_df, DIRS["tables"] / "all_split_metrics.parquet")
    return metrics_df



GROUP_DIMENSIONS = [
    "pore_regime",
    "density_regime",
    "topology_group",
    "topology_frequency_group",
    "geometry_family",
    "metal_family",
    "functional_family",
    "linker_family",
]


def load_all_prediction_files_for_target(target_col: str) -> pd.DataFrame:
    target_slug = slugify(target_col)
    files = sorted((DIRS["predictions"] / target_slug).glob("*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found for target: {target_col}")
    frames = [pd.read_csv(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def aggregate_main_benchmark_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["target", "model_name"], as_index=False)
        .agg(
            n_splits=("split_id", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values(["target", "rmse_mean"])
        .reset_index(drop=True)
    )
    return summary


def aggregate_group_level_errors(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_dim in GROUP_DIMENSIONS:
        if group_dim not in pred_df.columns:
            continue
        sub = pred_df[pred_df[group_dim].notna()].copy()
        group_sizes = sub[group_dim].value_counts()
        keep_groups = set(group_sizes[group_sizes >= MIN_GROUP_SIZE].index)
        sub = sub[sub[group_dim].isin(keep_groups)].copy()

        if sub.empty:
            continue

        grouped = sub.groupby(["target", "model_name", group_dim])
        for keys, g in grouped:
            target, model_name, group_value = keys
            rows.append(
                {
                    "target": target,
                    "model_name": model_name,
                    "group_dimension": group_dim,
                    "group_value": describe_group_label(group_value),
                    "n": int(len(g)),
                    "mae_g": float(g["abs_error"].mean()),
                    "rmse_g": float(np.sqrt(g["squared_error"].mean())),
                    "spearman_g": float(spearman_safe(g["y_true"], g["y_pred"])),
                    "elite_misclassification_rate_g": float(g["elite_misclassified"].mean()),
                    "elite_false_negative_rate_g": float(g["elite_false_negative"].mean()),
                    "abs_percentile_rank_error_g": float(g["abs_percentile_rank_error"].mean()),
                    "mean_novelty_nn1": float(g["nn_distance_1"].mean()) if "nn_distance_1" in g.columns else np.nan,
                    "mean_novelty_knn": float(g["nn_distance_mean_k"].mean()) if "nn_distance_mean_k" in g.columns else np.nan,
                    "mean_local_density": float(g["local_training_density"].mean()) if "local_training_density" in g.columns else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "model_name", "group_dimension", "mae_g"], ascending=[True, True, True, False])


def aggregate_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_dim in GROUP_DIMENSIONS:
        if group_dim not in df.columns:
            continue
        ser = df[group_dim].astype("object").fillna("missing")
        counts = ser.value_counts(dropna=False)
        for group_value, n in counts.items():
            rows.append(
                {
                    "group_dimension": group_dim,
                    "group_value": describe_group_label(group_value),
                    "n_structures": int(n),
                    "fraction_of_dataset": float(n / len(df)),
                }
            )
    out = pd.DataFrame(rows).sort_values(["group_dimension", "n_structures"], ascending=[True, False])
    return out


def build_hard_domain_rankings(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return group_df.copy()

    # Within each target/model, rank groups by multiple failure criteria
    out = group_df.copy()
    for metric in ["mae_g", "elite_misclassification_rate_g", "abs_percentile_rank_error_g", "mean_novelty_knn"]:
        out[f"rank_{metric}"] = out.groupby(["target", "model_name", "group_dimension"])[metric].rank(
            ascending=False, method="dense"
        )

    out["composite_hardness_score"] = (
        out["rank_mae_g"].fillna(0)
        + out["rank_elite_misclassification_rate_g"].fillna(0)
        + out["rank_abs_percentile_rank_error_g"].fillna(0)
        + out["rank_mean_novelty_knn"].fillna(0)
    )
    out = out.sort_values(["target", "model_name", "composite_hardness_score"], ascending=[True, True, True])
    return out


def aggregate_novelty_error_relationships(pred_df: pd.DataFrame) -> pd.DataFrame:
    novelty_cols = [c for c in ["nn_distance_1", "nn_distance_mean_k", "local_training_density"] if c in pred_df.columns]
    rows = []
    for target in pred_df["target"].dropna().unique():
        for model in pred_df["model_name"].dropna().unique():
            sub = pred_df[(pred_df["target"] == target) & (pred_df["model_name"] == model)].copy()
            if len(sub) < 20:
                continue
            for nov in novelty_cols:
                x = sub[nov]
                y = sub["abs_error"]
                mask = x.notna() & y.notna()
                if mask.sum() < 10:
                    continue
                rho = spearman_safe(x[mask], y[mask])
                rows.append(
                    {
                        "target": target,
                        "model_name": model,
                        "novelty_metric": nov,
                        "n": int(mask.sum()),
                        "spearman_novelty_vs_abs_error": rho,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_disagreement(pred_by_model: Dict[str, pd.DataFrame], target_col: str) -> pd.DataFrame:
    """
    Merge predictions from multiple models at sample level, then compute disagreement.
    """
    keys = ["id_norm", "filename", "split_id", "target"]
    base_cols = [
        "id_norm", "filename", "split_id", "target",
        "y_true", "Crystalnet", "pore_regime", "density_regime",
        "topology_group", "topology_frequency_group",
        "geometry_family", "metal_family", "functional_family", "linker_family",
        "Density", "Df", "Di", "Dif",
        "nn_distance_1", "nn_distance_mean_k", "local_training_density",
    ]

    merged = None
    for model_name, df_model in pred_by_model.items():
        keep = [c for c in base_cols if c in df_model.columns] + ["y_pred", "abs_error"]
        tmp = df_model[keep].copy().rename(
            columns={
                "y_pred": f"y_pred_{model_name}",
                "abs_error": f"abs_error_{model_name}",
            }
        )
        if merged is None:
            merged = tmp
        else:
            merge_cols = [c for c in keys if c in tmp.columns and c in merged.columns]
            merged = merged.merge(tmp, on=merge_cols, how="inner")

    if merged is None:
        return pd.DataFrame()

    pred_cols = [c for c in merged.columns if c.startswith("y_pred_")]
    err_cols = [c for c in merged.columns if c.startswith("abs_error_")]

    merged["prediction_spread_std"] = merged[pred_cols].std(axis=1)
    merged["prediction_spread_range"] = merged[pred_cols].max(axis=1) - merged[pred_cols].min(axis=1)
    merged["mean_abs_error_across_models"] = merged[err_cols].mean(axis=1)
    merged["worst_abs_error_across_models"] = merged[err_cols].max(axis=1)
    merged["local_trust_category"] = calculate_local_trust_category(
        abs_error=merged["mean_abs_error_across_models"],
        disagreement=merged["prediction_spread_std"],
    )

    return merged


def save_all_aggregated_outputs(df_master: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log("Aggregating benchmark-wide outputs.")

    coverage_df = aggregate_coverage_table(df_master)
    safe_to_csv(coverage_df, DIRS["tables"] / "table_coverage_statistics.csv", index=False)

    benchmark_df = aggregate_main_benchmark_summary(metrics_df)
    safe_to_csv(benchmark_df, DIRS["tables"] / "table_main_benchmark_summary.csv", index=False)

    all_predictions = []
    target_to_disagreement = {}
    for target_col in TARGET_COLUMNS:
        try:
            pred_df = load_all_prediction_files_for_target(target_col)
        except FileNotFoundError:
            log(f"No prediction files yet for target '{target_col}'. Skipping this target during aggregation.")
            continue
        safe_to_csv(pred_df, DIRS["tables"] / f"all_predictions__{slugify(target_col)}.csv", index=False)
        all_predictions.append(pred_df)

        pred_by_model = {}
        for model_name in MODEL_SPECS:
            model_sub = pred_df[pred_df["model_name"] == model_name].copy()
            if not model_sub.empty:
                pred_by_model[model_name] = model_sub
        dis_df = aggregate_disagreement(pred_by_model, target_col)
        target_to_disagreement[target_col] = dis_df
        safe_to_csv(dis_df, DIRS["tables"] / f"disagreement__{slugify(target_col)}.csv", index=False)

    if not all_predictions:
        raise RuntimeError("No prediction files were available for aggregation. Complete at least one model fit first.")
    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    safe_to_csv(all_pred_df, DIRS["tables"] / "all_predictions_all_targets.csv", index=False)
    safe_to_pickle(all_pred_df, DIRS["tables"] / "all_predictions_all_targets.pkl")
    maybe_to_parquet(all_pred_df, DIRS["tables"] / "all_predictions_all_targets.parquet")

    group_df = aggregate_group_level_errors(all_pred_df)
    safe_to_csv(group_df, DIRS["tables"] / "table_group_level_error_summary.csv", index=False)

    hard_df = build_hard_domain_rankings(group_df)
    safe_to_csv(hard_df, DIRS["tables"] / "table_hard_domain_rankings.csv", index=False)

    novelty_df = aggregate_novelty_error_relationships(all_pred_df)
    safe_to_csv(novelty_df, DIRS["tables"] / "table_novelty_error_relationships.csv", index=False)

    dis_rows = []
    for target_col, dis_df in target_to_disagreement.items():
        if dis_df.empty:
            continue
        for group_dim in GROUP_DIMENSIONS:
            if group_dim not in dis_df.columns:
                continue
            sub = dis_df[dis_df[group_dim].notna()].copy()
            counts = sub[group_dim].value_counts()
            keep = set(counts[counts >= MIN_GROUP_SIZE].index)
            sub = sub[sub[group_dim].isin(keep)]
            if sub.empty:
                continue
            grouped = sub.groupby(group_dim)
            for group_value, g in grouped:
                dis_rows.append(
                    {
                        "target": target_col,
                        "group_dimension": group_dim,
                        "group_value": describe_group_label(group_value),
                        "n": int(len(g)),
                        "prediction_spread_std_mean": float(g["prediction_spread_std"].mean()),
                        "prediction_spread_range_mean": float(g["prediction_spread_range"].mean()),
                        "mean_abs_error_across_models": float(g["mean_abs_error_across_models"].mean()),
                        "hard_and_unstable_fraction": float((g["local_trust_category"] == "hard_and_unstable").mean()),
                    }
                )
    disagreement_summary_df = pd.DataFrame(dis_rows)
    safe_to_csv(disagreement_summary_df, DIRS["tables"] / "table_agreement_disagreement_summary.csv", index=False)

    return {
        "coverage_df": coverage_df,
        "benchmark_df": benchmark_df,
        "all_pred_df": all_pred_df,
        "group_df": group_df,
        "hard_df": hard_df,
        "novelty_df": novelty_df,
        "disagreement_summary_df": disagreement_summary_df,
        "target_to_disagreement": target_to_disagreement,
        "benchmark_split_metrics_df": metrics_df,
    }



def render_table_image(df: pd.DataFrame, title: str, out_path: Path, max_rows: int = 18, max_cols: int = 8) -> None:
    """
    Save a simple publication-style table preview as PNG.
    This is not meant to replace LaTeX tables, but helps with quick manuscript assembly.
    """
    show = df.copy().head(max_rows)
    if show.shape[1] > max_cols:
        show = show.iloc[:, :max_cols]

    fig_h = max(2.5, 0.4 * (len(show) + 2))
    fig_w = max(8.0, 1.5 * show.shape[1])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=show.round(4).astype(str).values,
        colLabels=list(show.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    safe_tight_layout(fig)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def sanitize_matplotlib_text(text: object) -> str:
    """Return text that is safe for Matplotlib mathtext parsing.

    The previous composite version used a few labels with dollar-sign mathtext.
    On some Windows/Python/Matplotlib combinations, tight_layout can fail if a
    label is malformed or partly interpreted as mathtext. For production
    robustness, all figure labels are converted to plain text before layout and
    saving.
    """
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\\rho": "rho",
        "\\\\rho": "rho",
        "R^2": "R2",
        "CO_2": "CO2",
        "CH_4": "CH4",
        "": "",
    }
    for old_value, new_value in replacements.items():
        s = s.replace(old_value, new_value)
    return s


def sanitize_figure_text(fig: plt.Figure) -> None:
    """Remove fragile mathtext dollar signs from every text object in a figure."""
    for text_obj in fig.findobj(match=matplotlib.text.Text):
        try:
            text_obj.set_text(sanitize_matplotlib_text(text_obj.get_text()))
        except Exception:
            pass


def safe_tight_layout(fig: plt.Figure, rect=None) -> None:
    """Apply tight_layout safely; fall back to conservative spacing if needed."""
    sanitize_figure_text(fig)
    try:
        if rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=rect)
    except Exception as exc:
        log(f"Warning: tight_layout failed ({exc}). Using conservative subplot spacing instead.")
        try:
            if rect is None:
                fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.90, wspace=0.35, hspace=0.45)
            else:
                fig.subplots_adjust(
                    left=max(rect[0], 0.06),
                    right=min(rect[2], 0.96),
                    bottom=max(rect[1], 0.08),
                    top=min(rect[3], 0.92),
                    wspace=0.35,
                    hspace=0.45,
                )
        except Exception:
            pass


def savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    sanitize_figure_text(fig)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def save_figure_data(df: pd.DataFrame, figure_stem: str, panel_name: str = "data") -> None:
    """
    Save the exact numeric table used to make a figure or figure panel.
    Figure data go to results/figure_numeric_data/<figure>/<panel>.csv and .pkl.
    Large prediction-level outputs are already saved separately under results/tables/.
    """
    if df is None or df.empty:
        return
    safe_stem = slugify(figure_stem)
    safe_panel = slugify(panel_name)
    out_dir = DIRS["figure_data"] / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_to_csv(df.reset_index(drop=True), out_dir / f"{safe_panel}.csv", index=False)
    safe_to_pickle(df.reset_index(drop=True), out_dir / f"{safe_panel}.pkl")


def plot_workflow_figure(out_base: Path) -> None:
    """
    Figure 1: conceptual workflow figure.
    Since there is no external illustration dependency, we create a clean text-box flow diagram.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    boxes = [
        (0.05, 0.55, 0.18, 0.22, "Input CSV tables\nclean_data + clusters + topology"),
        (0.28, 0.55, 0.18, 0.22, "Identifier normalization\nand table merging"),
        (0.51, 0.55, 0.18, 0.22, "Interpretable partitions\npore / density / topology /\nchemistry-family proxies"),
        (0.74, 0.55, 0.18, 0.22, "Compact model panel\nRidge / RF / HGB"),
        (0.17, 0.15, 0.22, 0.22, "Per-sample outputs\npredictions, residuals,\nrank errors, elite labels"),
        (0.47, 0.15, 0.22, 0.22, "Failure analytics\nnovelty, disagreement,\ngroup-level metrics"),
        (0.77, 0.15, 0.18, 0.22, "Paper assets\nfailure maps, trust atlas,\nleaderboards, SI tables"),
    ]

    arrows = [
        ((0.23, 0.66), (0.28, 0.66)),
        ((0.46, 0.66), (0.51, 0.66)),
        ((0.69, 0.66), (0.74, 0.66)),
        ((0.83, 0.55), (0.83, 0.37)),
        ((0.39, 0.26), (0.47, 0.26)),
        ((0.69, 0.26), (0.77, 0.26)),
        ((0.60, 0.55), (0.28, 0.37)),
    ]

    workflow_df = pd.DataFrame(boxes, columns=["x", "y", "width", "height", "label"])
    arrow_df = pd.DataFrame([{"start_x": a[0][0], "start_y": a[0][1], "end_x": a[1][0], "end_y": a[1][1]} for a in arrows])
    save_figure_data(workflow_df, out_base.name, "workflow_boxes")
    save_figure_data(arrow_df, out_base.name, "workflow_arrows")

    for (x, y, w, h, text) in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4))

    ax.set_title("Failure-map workflow: from merged CSV tables to local reliability maps", fontsize=14, pad=12)
    savefig(fig, out_base)


def plot_domain_resolved_heatmap(pred_df: pd.DataFrame, target_col: str, model_name: str, out_base: Path) -> None:
    """
    Figure 2: Domain-resolved error heatmap.
    Default pairing: pore_regime x topology_frequency_group or metal_family.
    """
    sub = pred_df[(pred_df["target"] == target_col) & (pred_df["model_name"] == model_name)].copy()
    if sub.empty:
        return

    # Prefer metal_family if not too many categories, else topology_frequency_group
    y_dim = "metal_family"
    counts = sub[y_dim].value_counts()
    top_levels = counts.head(MAX_CATEGORIES_FOR_PLOTS).index
    if len(top_levels) < 3:
        y_dim = "topology_frequency_group"
        top_levels = sub[y_dim].value_counts().head(MAX_CATEGORIES_FOR_PLOTS).index

    plot_df = sub[sub[y_dim].isin(top_levels)].copy()
    heat = plot_df.pivot_table(
        index=y_dim,
        columns="pore_regime",
        values="abs_error",
        aggfunc="mean",
    )
    heat_long = heat.reset_index().melt(id_vars=[y_dim], var_name="pore_regime", value_name="mean_abs_error")
    save_figure_data(heat_long, out_base.name, "domain_resolved_error_heatmap")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(heat))))
    im = ax.imshow(heat.fillna(np.nan).to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title(f"Domain-resolved mean absolute error\n{model_name.upper()} | {target_col}", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean absolute error")
    safe_tight_layout(fig)
    savefig(fig, out_base)


def plot_novelty_vs_error(pred_df: pd.DataFrame, target_col: str, model_name: str, out_base: Path) -> None:
    """
    Figure 3: novelty-versus-error map.
    """
    sub = pred_df[(pred_df["target"] == target_col) & (pred_df["model_name"] == model_name)].copy()
    if sub.empty or "nn_distance_mean_k" not in sub.columns:
        return

    fig_data = sub[["target", "model_name", "id_norm", "nn_distance_mean_k", "abs_error", "elite_misclassified"]].copy()
    save_figure_data(fig_data, out_base.name, "novelty_vs_error_points")

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        sub["nn_distance_mean_k"],
        sub["abs_error"],
        gridsize=45,
        mincnt=1,
    )
    ax.set_xlabel("Novelty proxy: mean kNN distance to training set")
    ax.set_ylabel("Absolute prediction error")
    ax.set_title(f"Novelty-versus-error map\n{model_name.upper()} | {target_col}")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")
    safe_tight_layout(fig)
    savefig(fig, out_base)


def plot_hard_domain_leaderboard(hard_df: pd.DataFrame, target_col: str, model_name: str, out_base: Path) -> None:
    """
    Figure 4: hardest groups by composite hardness score.
    """
    sub = hard_df[(hard_df["target"] == target_col) & (hard_df["model_name"] == model_name)].copy()
    if sub.empty:
        return

    show = (
        sub.sort_values("composite_hardness_score", ascending=True)
        .head(TOP_LEADERBOARD_N)
        .copy()
    )
    show["label"] = show["group_dimension"].astype(str) + " | " + show["group_value"].astype(str)
    save_figure_data(show, out_base.name, "hard_domain_leaderboard")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(show))))
    ax.barh(show["label"], show["mae_g"])
    ax.invert_yaxis()
    ax.set_xlabel("Group MAE")
    ax.set_title(f"Hard-domain leaderboard\n{model_name.upper()} | {target_col}")
    safe_tight_layout(fig)
    savefig(fig, out_base)


def plot_disagreement_map(dis_df: pd.DataFrame, target_col: str, out_base: Path) -> None:
    """
    Figure 5: model disagreement in descriptor space.
    """
    if dis_df.empty:
        return

    x_col, y_col = TRUST_MAP_AXES
    sub = dis_df[[x_col, y_col, "prediction_spread_std"]].dropna().copy()
    if sub.empty:
        return
    save_figure_data(sub, out_base.name, "model_disagreement_points")

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(sub[x_col], sub[y_col], C=sub["prediction_spread_std"], reduce_C_function=np.mean, gridsize=35)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Model disagreement map\n{target_col}")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Mean prediction spread (std across models)")
    safe_tight_layout(fig)
    savefig(fig, out_base)


def plot_local_trust_atlas(dis_df: pd.DataFrame, target_col: str, out_base: Path) -> None:
    """
    Figure 6: local trust atlas in Df-Density space using dominant trust category per 2D bin.
    """
    if dis_df.empty:
        return

    x_col, y_col = TRUST_MAP_AXES
    sub = dis_df[[x_col, y_col, "local_trust_category"]].dropna().copy()
    if sub.empty:
        return

    # Bin the continuous space
    sub["x_bin"] = pd.qcut(sub[x_col], q=12, duplicates="drop")
    sub["y_bin"] = pd.qcut(sub[y_col], q=12, duplicates="drop")

    grouped = sub.groupby(["y_bin", "x_bin"])
    rows = []
    for (yb, xb), g in grouped:
        if len(g) < TRUST_BIN_MIN_COUNT:
            continue
        dominant = g["local_trust_category"].value_counts().idxmax()
        rows.append({"y_bin": str(yb), "x_bin": str(xb), "dominant": dominant, "n": len(g)})

    atlas = pd.DataFrame(rows)
    if atlas.empty:
        return
    save_figure_data(atlas, out_base.name, "local_trust_atlas_bins")

    x_levels = atlas["x_bin"].drop_duplicates().tolist()
    y_levels = atlas["y_bin"].drop_duplicates().tolist()

    category_codes = {
        "easy_and_stable": 0,
        "intermediate": 1,
        "ambiguous_or_model_sensitive": 2,
        "hard_but_consistent": 3,
        "hard_and_unstable": 4,
        "missing": 5,
    }
    Z = np.full((len(y_levels), len(x_levels)), np.nan)
    for _, r in atlas.iterrows():
        yi = y_levels.index(r["y_bin"])
        xi = x_levels.index(r["x_bin"])
        Z[yi, xi] = category_codes.get(r["dominant"], np.nan)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(Z, aspect="auto")
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(y_levels)))
    ax.set_yticklabels(y_levels, fontsize=7)
    ax.set_xlabel(f"{x_col} quantile bins")
    ax.set_ylabel(f"{y_col} quantile bins")
    ax.set_title(f"Local trust atlas\n{target_col}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(list(category_codes.values()))
    cbar.set_ticklabels(list(category_codes.keys()))

    safe_tight_layout(fig)
    savefig(fig, out_base)



def make_bootstrap_ci(values: pd.Series, n_boot: int = 1000, seed: int = BASE_RANDOM_SEED) -> Tuple[float, float, float]:
    vals = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_statistical_reliability_composite(
    metrics_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    dis_df: pd.DataFrame,
    target_col: str,
    out_base: Path,
) -> None:
    """
    Create the statistical reliability and screening-risk diagnostic composite.

    The panels summarize bootstrap uncertainty intervals, novelty--risk
    association, local trust categories, and risk stratification across
    descriptor-space novelty strata.
    """
    if metrics_df.empty or dis_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    panel_a_rows = []
    msub = metrics_df[metrics_df["target"] == target_col].copy()
    for model_name, g in msub.groupby("model_name"):
        mean, lo, hi = make_bootstrap_ci(g["mae"], seed=BASE_RANDOM_SEED + len(model_name))
        panel_a_rows.append({"model_name": model_name, "mae_mean": mean, "mae_ci_low": lo, "mae_ci_high": hi})
    panel_a = pd.DataFrame(panel_a_rows).sort_values("mae_mean") if panel_a_rows else pd.DataFrame()
    save_figure_data(panel_a, out_base.name, "panel_a_bootstrap_mae_ci")
    if not panel_a.empty:
        x = np.arange(len(panel_a))
        ax_a.bar(x, panel_a["mae_mean"].to_numpy())
        ax_a.errorbar(x, panel_a["mae_mean"].to_numpy(),
                      yerr=[panel_a["mae_mean"].to_numpy()-panel_a["mae_ci_low"].to_numpy(),
                            panel_a["mae_ci_high"].to_numpy()-panel_a["mae_mean"].to_numpy()],
                      fmt="none", capsize=4)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(panel_a["model_name"].str.upper())
    ax_a.set_ylabel("MAE")
    ax_a.set_title("A. Split-level MAE with bootstrap 95% CI")

    panel_b = novelty_df[
        (novelty_df["target"] == target_col) &
        (novelty_df["novelty_metric"] == "nn_distance_mean_k")
    ].copy() if not novelty_df.empty else pd.DataFrame()
    save_figure_data(panel_b, out_base.name, "panel_b_novelty_error_correlation")
    if not panel_b.empty:
        panel_b = panel_b.sort_values("spearman_novelty_vs_abs_error")
        x = np.arange(len(panel_b))
        ax_b.bar(x, panel_b["spearman_novelty_vs_abs_error"].to_numpy())
        ax_b.axhline(0, linewidth=1)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(panel_b["model_name"].str.upper())
    ax_b.set_ylabel("Spearman rho")
    ax_b.set_title("B. Novelty-error association")

    panel_c = (dis_df["local_trust_category"].value_counts(normalize=True)
               .rename_axis("local_trust_category").reset_index(name="fraction"))
    save_figure_data(panel_c, out_base.name, "panel_c_local_trust_fractions")
    if not panel_c.empty:
        ax_c.barh(panel_c["local_trust_category"], panel_c["fraction"])
        ax_c.invert_yaxis()
    ax_c.set_xlabel("Fraction of test predictions")
    ax_c.set_title("C. Local trust category distribution")

    risk = dis_df[["nn_distance_mean_k", "mean_abs_error_across_models"]].dropna().copy()
    if len(risk) >= 50:
        risk["novelty_quintile"] = pd.qcut(risk["nn_distance_mean_k"], q=5, labels=False, duplicates="drop")
        panel_d = risk.groupby("novelty_quintile", as_index=False).agg(
            n=("mean_abs_error_across_models", "size"),
            novelty_mean=("nn_distance_mean_k", "mean"),
            mean_abs_error=("mean_abs_error_across_models", "mean"),
        )
    else:
        panel_d = pd.DataFrame()
    save_figure_data(panel_d, out_base.name, "panel_d_novelty_quintile_risk")
    if not panel_d.empty:
        ax_d.plot(panel_d["novelty_quintile"], panel_d["mean_abs_error"], marker="o")
        ax_d.set_xticks(panel_d["novelty_quintile"].to_numpy())
        ax_d.set_xlabel("Novelty quintile")
    ax_d.set_ylabel("Mean absolute error across models")
    ax_d.set_title("D. Reliability risk across novelty strata")

    fig.suptitle(f"Statistical reliability and local-risk diagnostics\n{target_col}", fontsize=14)
    safe_tight_layout(fig, rect=[0, 0, 1, 0.95])
    savefig(fig, out_base)





TARGET_SHORT_LABELS = {
    "uptake(mmol/g) CO2 at 0.015 bar": "CO2 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar": "CO2 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar": "CH4 5.8 bar",
    "uptake(mmol/g) methane at 65 bar": "CH4 65 bar",
}

TARGET_FILE_LABELS = {
    "uptake(mmol/g) CO2 at 0.015 bar": "co2_0015_bar",
    "uptake(mmol/g) CO2 at 0.15 bar": "co2_015_bar",
    "uptake(mmol/g) methane at 5.8 bar": "ch4_58_bar",
    "uptake(mmol/g) methane at 65 bar": "ch4_65_bar",
}

TRUST_CATEGORY_ORDER = [
    "easy_and_stable",
    "intermediate",
    "ambiguous_or_model_sensitive",
    "hard_but_consistent",
    "hard_and_unstable",
]

PRETTY_TRUST_LABELS = {
    "easy_and_stable": "Easy + stable",
    "intermediate": "Intermediate",
    "ambiguous_or_model_sensitive": "Model-sensitive",
    "hard_but_consistent": "Hard + consistent",
    "hard_and_unstable": "Hard + unstable",
}


def short_target_label(target_col: str) -> str:
    return TARGET_SHORT_LABELS.get(target_col, target_col)


def file_target_label(target_col: str) -> str:
    return TARGET_FILE_LABELS.get(target_col, slugify(target_col))


def pretty_model(model_name: str) -> str:
    return str(model_name).upper()


def clean_axis_label(value: object, max_len: int = 34) -> str:
    text = str(value)
    replacements = {
        "ultramicroporous_or_tight": "ultramicroporous/tight",
        "high_constriction_contrast": "high constriction",
        "topology_frequency_group": "topology frequency",
        "geometry_family": "geometry family",
        "density_regime": "density",
        "pore_regime": "pore",
        "other_geometry_families": "other geometry families",
        "other_metal_families": "other metal families",
        "other_functional_families": "other functional families",
        "other_linker_families": "other linker families",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("__", " + ").replace("_", " ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08, 1.04, label, transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="bottom", ha="left"
    )


def plot_matrix_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cbar_label: str,
                        fmt: str = ".2f", annotate: bool = True) -> None:
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([clean_axis_label(c, 22) for c in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([clean_axis_label(i, 28) for i in matrix.index])
    ax.set_title(title, fontsize=11)
    if annotate and matrix.shape[0] * matrix.shape[1] <= 40:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)


def select_primary_target_from_results(agg: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score targets for main-text suitability and save the decision table.

    The score is deliberately transparent rather than automatic manuscript truth:
    it rewards (i) good nonlinear performance, (ii) clear novelty-risk association,
    (iii) strong hard-domain contrast, and (iv) strong process relevance to CO2 capture.
    """
    benchmark = agg.get("benchmark_df", pd.DataFrame()).copy()
    novelty = agg.get("novelty_df", pd.DataFrame()).copy()
    group_df = agg.get("group_df", pd.DataFrame()).copy()

    rows = []
    for target in TARGET_COLUMNS:
        b = benchmark[benchmark["target"] == target]
        if b.empty:
            continue
        best_mae = b["mae_mean"].min()
        best_r2 = b["r2_mean"].max()
        ridge_mae = b.loc[b["model_name"] == "ridge", "mae_mean"].mean()
        nonlinear_gain = (ridge_mae - best_mae) / ridge_mae if np.isfinite(ridge_mae) and ridge_mae > 0 else np.nan

        n = novelty[(novelty["target"] == target) & (novelty["novelty_metric"] == "nn_distance_mean_k")]
        novelty_signal = n["spearman_novelty_vs_abs_error"].abs().mean() if not n.empty else np.nan

        g = group_df[group_df["target"] == target]
        hard_contrast = np.nan
        if not g.empty:
            q90 = g["mae_g"].quantile(0.90)
            q50 = g["mae_g"].quantile(0.50)
            hard_contrast = q90 / q50 if q50 and np.isfinite(q50) else np.nan

        process_relevance = 1.0 if "CO2 at 0.15" in target else (0.85 if "CO2 at 0.015" in target else 0.65)
        visual_story_score = (
            0.30 * np.nan_to_num(best_r2, nan=0.0) +
            0.25 * np.nan_to_num(novelty_signal, nan=0.0) +
            0.20 * min(np.nan_to_num(hard_contrast, nan=0.0), 3.0) / 3.0 +
            0.15 * np.nan_to_num(nonlinear_gain, nan=0.0) +
            0.10 * process_relevance
        )
        rows.append({
            "target": target,
            "recommended_main_text": target == PRIMARY_TARGET,
            "short_label": short_target_label(target),
            "best_mae": best_mae,
            "best_r2": best_r2,
            "ridge_to_best_relative_mae_gain": nonlinear_gain,
            "mean_abs_novelty_error_spearman": novelty_signal,
            "hard_domain_contrast_q90_over_q50": hard_contrast,
            "process_relevance_weight": process_relevance,
            "story_score": visual_story_score,
            "recommendation_note": (
                "Use as the main-text anchor: CO2 capture-relevant, chemically interpretable, and shows clear local failure domains."
                if target == PRIMARY_TARGET else
                "Keep as SI/robustness target unless a target-specific journal narrative requires it."
            ),
        })
    decision = pd.DataFrame(rows).sort_values("story_score", ascending=False)
    safe_to_csv(decision, DIRS["tables"] / "table_main_text_target_selection_rationale.csv", index=False)
    save_figure_data(decision, "Figure_1_target_selection_and_workflow", "target_selection_rationale")
    return decision


def plot_workflow_and_target_selection_composite(agg: Dict[str, pd.DataFrame], out_base: Path) -> None:
    """Composite Figure 1: why this target anchors the main text + workflow."""
    decision = select_primary_target_from_results(agg)
    benchmark = agg.get("benchmark_df", pd.DataFrame()).copy()

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.38)
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_perf = fig.add_subplot(gs[1, 1])

    # workflow panel, redrawn in a more compact form
    ax_flow.axis("off")
    boxes = [
        (0.03, 0.78, 0.35, 0.13, "CSV inputs\ngeometry + topology + cluster tables"),
        (0.55, 0.78, 0.35, 0.13, "Clean merge\nID normalization + safe categories"),
        (0.03, 0.52, 0.35, 0.13, "Compact models\nRidge / RF / HGB"),
        (0.55, 0.52, 0.35, 0.13, "Per-sample diagnostics\nerror + rank + novelty"),
        (0.03, 0.26, 0.35, 0.13, "Domain maps\npore / density / topology / chemistry"),
        (0.55, 0.26, 0.35, 0.13, "Trust atlas\nwhere predictions are locally reliable"),
    ]
    for x, y, w, h, txt in boxes:
        ax_flow.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.4))
        ax_flow.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=10)
    arrows = [((0.38, 0.845), (0.55, 0.845)), ((0.72, 0.78), (0.20, 0.65)),
              ((0.38, 0.585), (0.55, 0.585)), ((0.72, 0.52), (0.20, 0.39)),
              ((0.38, 0.325), (0.55, 0.325))]
    for start, end in arrows:
        ax_flow.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.3))
    ax_flow.set_title("A. Failure-map workflow", fontsize=12)
    save_figure_data(pd.DataFrame(boxes, columns=["x", "y", "width", "height", "label"]), out_base.name, "panel_a_workflow_boxes")

    # target-selection score
    if not decision.empty:
        dec = decision.sort_values("story_score", ascending=True)
        labels = [short_target_label(t) for t in dec["target"]]
        ax_score.barh(labels, dec["story_score"])
        for y, is_primary in enumerate(dec["target"] == PRIMARY_TARGET):
            if is_primary:
                ax_score.text(dec["story_score"].iloc[y] + 0.01, y, "main text", va="center", fontsize=9)
        ax_score.set_xlabel("Transparent story score")
    ax_score.set_title("B. Main-text anchor choice")

    # model performance for all targets
    if not benchmark.empty:
        perf = benchmark.pivot_table(index="target", columns="model_name", values="r2_mean", aggfunc="mean")
        perf = perf.reindex(TARGET_COLUMNS).rename(index=short_target_label, columns=pretty_model)
        plot_matrix_heatmap(ax_perf, perf, "C. Predictive performance across tasks", "Mean R2", fmt=".2f", annotate=True)

    fig.suptitle("Main-text focus: CO2 0.15 bar as the clearest failure-map anchor; CH4 65 bar retained as a strong robustness case", fontsize=14)
    savefig(fig, out_base)


def plot_benchmark_reliability_landscape_composite(agg: Dict[str, pd.DataFrame], out_base: Path) -> None:
    """Composite Figure 2: model performance and reliability signals across all targets."""
    benchmark = agg.get("benchmark_df", pd.DataFrame()).copy()
    novelty = agg.get("novelty_df", pd.DataFrame()).copy()
    target_to_disagreement = agg.get("target_to_disagreement", {})
    all_pred = agg.get("all_pred_df", pd.DataFrame()).copy()
    if benchmark.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A: normalized MAE by task/model using target IQR for comparability
    iqr_rows = []
    if not all_pred.empty:
        for target, g in all_pred.groupby("target"):
            vals = g["y_true"].dropna()
            iqr = vals.quantile(0.75) - vals.quantile(0.25)
            if not np.isfinite(iqr) or iqr <= 0:
                iqr = vals.std(ddof=0)
            iqr_rows.append({"target": target, "target_iqr": float(iqr)})
    iqr_df = pd.DataFrame(iqr_rows)
    bench_norm = benchmark.merge(iqr_df, on="target", how="left")
    bench_norm["mae_over_iqr"] = bench_norm["mae_mean"] / bench_norm["target_iqr"]
    mae_mat = bench_norm.pivot_table(index="target", columns="model_name", values="mae_over_iqr", aggfunc="mean")
    mae_mat = mae_mat.reindex(TARGET_COLUMNS).rename(index=short_target_label, columns=pretty_model)
    save_figure_data(bench_norm, out_base.name, "panel_a_normalized_mae")
    plot_matrix_heatmap(ax_a, mae_mat, "A. Error normalized by target IQR", "MAE / target IQR", fmt=".2f", annotate=True)

    # B: nonlinear gain over Ridge
    gain_rows = []
    for target, g in benchmark.groupby("target"):
        ridge = g.loc[g["model_name"] == "ridge", "mae_mean"]
        if ridge.empty or ridge.iloc[0] <= 0:
            continue
        ridge_val = ridge.iloc[0]
        for _, r in g.iterrows():
            gain_rows.append({"target": target, "model_name": r["model_name"], "relative_mae_gain_vs_ridge": (ridge_val - r["mae_mean"]) / ridge_val})
    gain_df = pd.DataFrame(gain_rows)
    gain_mat = gain_df.pivot_table(index="target", columns="model_name", values="relative_mae_gain_vs_ridge", aggfunc="mean")
    gain_mat = gain_mat.reindex(TARGET_COLUMNS).rename(index=short_target_label, columns=pretty_model)
    save_figure_data(gain_df, out_base.name, "panel_b_nonlinear_gain")
    plot_matrix_heatmap(ax_b, gain_mat, "B. Gain relative to Ridge", "Relative MAE gain", fmt=".2f", annotate=True)

    # C: novelty-error correlations
    nov = novelty[novelty["novelty_metric"] == "nn_distance_mean_k"].copy() if not novelty.empty else pd.DataFrame()
    if not nov.empty:
        nov_mat = nov.pivot_table(index="target", columns="model_name", values="spearman_novelty_vs_abs_error", aggfunc="mean")
        nov_mat = nov_mat.reindex(TARGET_COLUMNS).rename(index=short_target_label, columns=pretty_model)
        save_figure_data(nov, out_base.name, "panel_c_novelty_error_spearman")
        plot_matrix_heatmap(ax_c, nov_mat, "C. Novelty-error association", "Spearman rho", fmt=".2f", annotate=True)
    else:
        ax_c.axis("off")

    # D: trust category distribution across targets
    trust_rows = []
    for target, dis_df in target_to_disagreement.items():
        if dis_df is None or dis_df.empty or "local_trust_category" not in dis_df.columns:
            continue
        vc = dis_df["local_trust_category"].value_counts(normalize=True)
        for cat in TRUST_CATEGORY_ORDER:
            trust_rows.append({"target": target, "category": cat, "fraction": float(vc.get(cat, 0.0))})
    trust_df = pd.DataFrame(trust_rows)
    save_figure_data(trust_df, out_base.name, "panel_d_trust_category_fractions")
    if not trust_df.empty:
        pivot = trust_df.pivot_table(index="target", columns="category", values="fraction", fill_value=0.0)
        pivot = pivot.reindex(TARGET_COLUMNS)
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for cat in TRUST_CATEGORY_ORDER:
            vals = pivot.get(cat, pd.Series(0, index=pivot.index)).to_numpy(dtype=float)
            ax_d.bar(x, vals, bottom=bottom, label=PRETTY_TRUST_LABELS.get(cat, cat))
            bottom += vals
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([short_target_label(t) for t in pivot.index], rotation=25, ha="right")
        ax_d.set_ylabel("Fraction of test predictions")
        ax_d.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax_d.set_title("D. Local trust category balance")

    for label, ax in zip(["A", "B", "C", "D"], axes.ravel()):
        add_panel_label(ax, label)
    fig.suptitle("Benchmark reliability landscape across adsorption targets", fontsize=15)
    safe_tight_layout(fig, rect=[0, 0, 0.90, 0.95])
    savefig(fig, out_base)


def build_primary_domain_matrix(pred_df: pd.DataFrame, target_col: str, model_name: str) -> Tuple[pd.DataFrame, str]:
    sub = pred_df[(pred_df["target"] == target_col) & (pred_df["model_name"] == model_name)].copy()
    if sub.empty:
        return pd.DataFrame(), "metal_family"
    y_dim = "metal_family"
    counts = sub[y_dim].value_counts()
    top_levels = counts.head(14).index
    plot_df = sub[sub[y_dim].isin(top_levels)].copy()
    heat = plot_df.pivot_table(index=y_dim, columns="pore_regime", values="abs_error", aggfunc="mean")
    return heat, y_dim


def plot_primary_failure_atlas_composite(agg: Dict[str, pd.DataFrame], target_col: str, model_name: str, out_base: Path) -> None:
    """Composite Figure 3: the main failure map for the selected CO2 target."""
    all_pred = agg.get("all_pred_df", pd.DataFrame()).copy()
    hard_df = agg.get("hard_df", pd.DataFrame()).copy()
    novelty_df = agg.get("novelty_df", pd.DataFrame()).copy()
    if all_pred.empty:
        return
    sub = all_pred[(all_pred["target"] == target_col) & (all_pred["model_name"] == model_name)].copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    heat, y_dim = build_primary_domain_matrix(all_pred, target_col, model_name)
    if not heat.empty:
        heat_long = heat.reset_index().melt(id_vars=[y_dim], var_name="pore_regime", value_name="mean_abs_error")
        save_figure_data(heat_long, out_base.name, "panel_a_domain_error_heatmap")
        plot_matrix_heatmap(ax_a, heat.rename(index=lambda x: clean_axis_label(x, 22), columns=lambda x: clean_axis_label(x, 18)),
                            "A. Domain-resolved MAE", "Mean absolute error", fmt=".2f", annotate=False)

    # novelty map: hexbin for density plus top-error outlier envelope
    fig_data = sub[["nn_distance_mean_k", "abs_error", "elite_misclassified", "Df", "Density"]].dropna().copy()
    save_figure_data(fig_data, out_base.name, "panel_b_novelty_error_points")
    hb = ax_b.hexbin(fig_data["nn_distance_mean_k"], fig_data["abs_error"], gridsize=42, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax_b, shrink=0.85)
    cbar.set_label("Count")
    ax_b.set_xlabel("Novelty: mean kNN distance")
    ax_b.set_ylabel("Absolute error")
    rho = spearman_safe(fig_data["nn_distance_mean_k"], fig_data["abs_error"])
    ax_b.set_title(f"B. Novelty-error map (rho={rho:.2f})")

    # hard-domain leaderboard
    hard = hard_df[(hard_df["target"] == target_col) & (hard_df["model_name"] == model_name)].copy()
    hard = hard.sort_values("mae_g", ascending=False).head(10)
    hard["label"] = hard["group_dimension"].astype(str) + " | " + hard["group_value"].astype(str)
    save_figure_data(hard, out_base.name, "panel_c_hard_domain_leaderboard")
    if not hard.empty:
        hard_plot = hard.sort_values("mae_g", ascending=True)
        ax_c.barh([clean_axis_label(v, 36) for v in hard_plot["label"]], hard_plot["mae_g"])
        ax_c.set_xlabel("Group MAE")
    ax_c.set_title("C. Hardest interpretable domains")

    # novelty-stratified error by model
    rows = []
    for m, g in all_pred[all_pred["target"] == target_col].groupby("model_name"):
        temp = g[["nn_distance_mean_k", "abs_error"]].dropna().copy()
        if len(temp) < 50:
            continue
        temp["novelty_quintile"] = pd.qcut(temp["nn_distance_mean_k"], q=5, labels=False, duplicates="drop")
        tmp = temp.groupby("novelty_quintile", as_index=False).agg(mean_abs_error=("abs_error", "mean"), n=("abs_error", "size"))
        tmp["model_name"] = m
        rows.append(tmp)
    risk_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    save_figure_data(risk_df, out_base.name, "panel_d_error_by_novelty_quintile")
    if not risk_df.empty:
        for m, g in risk_df.groupby("model_name"):
            ax_d.plot(g["novelty_quintile"], g["mean_abs_error"], marker="o", label=pretty_model(m))
        ax_d.set_xlabel("Novelty quintile")
        ax_d.set_ylabel("Mean absolute error")
        ax_d.legend(fontsize=9)
    ax_d.set_title("D. Risk rises toward novel regions")

    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)
    fig.suptitle(f"Failure atlas for {short_target_label(target_col)} ({pretty_model(model_name)})", fontsize=16)
    safe_tight_layout(fig, rect=[0, 0, 1, 0.95])
    savefig(fig, out_base)


def plot_trust_and_disagreement_composite(agg: Dict[str, pd.DataFrame], target_col: str, out_base: Path) -> None:
    """Composite Figure 4: local trust atlas, disagreement and screening consequences."""
    dis_df = agg.get("target_to_disagreement", {}).get(target_col, pd.DataFrame()).copy()
    if dis_df.empty:
        return

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # disagreement map
    pts = dis_df[["Df", "Density", "prediction_spread_std", "mean_abs_error_across_models"]].dropna().copy()
    save_figure_data(pts, out_base.name, "panel_a_disagreement_points")
    hb = ax_a.hexbin(pts["Df"], pts["Density"], C=pts["prediction_spread_std"], reduce_C_function=np.mean, gridsize=34)
    fig.colorbar(hb, ax=ax_a, shrink=0.85).set_label("Mean prediction spread")
    ax_a.set_xlabel("PLD, Df")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A. Where models disagree")

    # trust atlas rendered directly
    atlas_source = dis_df[["Df", "Density", "local_trust_category"]].dropna().copy()
    atlas_source["x_bin"] = pd.qcut(atlas_source["Df"], q=12, duplicates="drop")
    atlas_source["y_bin"] = pd.qcut(atlas_source["Density"], q=12, duplicates="drop")
    rows = []
    for (yb, xb), g in atlas_source.groupby(["y_bin", "x_bin"]):
        if len(g) < TRUST_BIN_MIN_COUNT:
            continue
        rows.append({"y_bin": str(yb), "x_bin": str(xb), "dominant": g["local_trust_category"].value_counts().idxmax(), "n": len(g)})
    atlas = pd.DataFrame(rows)
    save_figure_data(atlas, out_base.name, "panel_b_trust_atlas_bins")
    if not atlas.empty:
        x_levels = atlas["x_bin"].drop_duplicates().tolist()
        y_levels = atlas["y_bin"].drop_duplicates().tolist()
        codes = {c: i for i, c in enumerate(TRUST_CATEGORY_ORDER)}
        Z = np.full((len(y_levels), len(x_levels)), np.nan)
        for _, r in atlas.iterrows():
            Z[y_levels.index(r["y_bin"]), x_levels.index(r["x_bin"])] = codes.get(r["dominant"], np.nan)
        im = ax_b.imshow(Z, aspect="auto")
        ax_b.set_xticks(np.arange(len(x_levels)))
        ax_b.set_xticklabels([clean_axis_label(v, 12) for v in x_levels], rotation=90, fontsize=6)
        ax_b.set_yticks(np.arange(len(y_levels)))
        ax_b.set_yticklabels([clean_axis_label(v, 12) for v in y_levels], fontsize=6)
        cb = fig.colorbar(im, ax=ax_b, shrink=0.85)
        cb.set_ticks(list(codes.values()))
        cb.set_ticklabels([PRETTY_TRUST_LABELS[c] for c in TRUST_CATEGORY_ORDER])
    ax_b.set_xlabel("PLD quantile bins")
    ax_b.set_ylabel("Density quantile bins")
    ax_b.set_title("B. Local trust atlas")

    # category fractions
    vc = dis_df["local_trust_category"].value_counts(normalize=True)
    frac_df = pd.DataFrame({"category": TRUST_CATEGORY_ORDER, "fraction": [float(vc.get(c, 0.0)) for c in TRUST_CATEGORY_ORDER]})
    save_figure_data(frac_df, out_base.name, "panel_c_trust_fractions")
    ax_c.barh([PRETTY_TRUST_LABELS[c] for c in frac_df["category"]], frac_df["fraction"])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Fraction of test predictions")
    ax_c.set_title("C. How much of the map is risky?")

    # false-negative risk among elite candidates by trust category
    model_err_cols = [c for c in dis_df.columns if c.startswith("abs_error_")]
    elite_cols = [c for c in dis_df.columns if c.startswith("y_pred_")]
    # In disagreement table, elite_misclassified is not present after merge; use mean error by trust category instead.
    risk = dis_df.groupby("local_trust_category", as_index=False).agg(
        n=("mean_abs_error_across_models", "size"),
        mean_abs_error=("mean_abs_error_across_models", "mean"),
        worst_abs_error=("worst_abs_error_across_models", "mean"),
        prediction_spread=("prediction_spread_std", "mean"),
    )
    risk["category"] = pd.Categorical(risk["local_trust_category"], TRUST_CATEGORY_ORDER, ordered=True)
    risk = risk.sort_values("category")
    save_figure_data(risk, out_base.name, "panel_d_risk_by_trust_category")
    ax_d.plot([PRETTY_TRUST_LABELS.get(c, c) for c in risk["local_trust_category"]], risk["mean_abs_error"], marker="o", label="Mean error")
    ax_d.plot([PRETTY_TRUST_LABELS.get(c, c) for c in risk["local_trust_category"]], risk["prediction_spread"], marker="s", label="Model spread")
    ax_d.set_xticklabels([PRETTY_TRUST_LABELS.get(c, c) for c in risk["local_trust_category"]], rotation=30, ha="right")
    ax_d.set_ylabel("Mean value")
    ax_d.legend(fontsize=9)
    ax_d.set_title("D. Error and disagreement co-localize")

    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)
    fig.suptitle(f"From model disagreement to local trust for {short_target_label(target_col)}", fontsize=16)
    safe_tight_layout(fig, rect=[0, 0, 1, 0.95])
    savefig(fig, out_base)


def plot_recurring_failure_motifs_composite(agg: Dict[str, pd.DataFrame], out_base: Path) -> None:
    """Composite Figure 5: which hard domains recur across targets and models."""
    hard_df = agg.get("hard_df", pd.DataFrame()).copy()
    group_df = agg.get("group_df", pd.DataFrame()).copy()
    if hard_df.empty or group_df.empty:
        return

    # Define high-risk groups as the top decile within each target/model.
    temp = group_df.copy()
    temp["risk_quantile"] = temp.groupby(["target", "model_name"])["mae_g"].rank(pct=True)
    temp["group_key"] = temp["group_dimension"].astype(str) + " | " + temp["group_value"].astype(str)
    recurrent = temp[temp["risk_quantile"] >= 0.90].groupby("group_key", as_index=False).agg(
        n_target_model_cases=("target", "size"),
        n_targets=("target", "nunique"),
        mean_mae=("mae_g", "mean"),
        mean_rank_error=("abs_percentile_rank_error_g", "mean"),
        mean_elite_misclassification=("elite_misclassification_rate_g", "mean"),
    ).sort_values(["n_targets", "n_target_model_cases", "mean_mae"], ascending=False).head(15)
    save_figure_data(recurrent, out_base.name, "panel_a_recurrent_failure_motifs")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax_a, ax_b = axes
    if not recurrent.empty:
        plot = recurrent.sort_values("n_target_model_cases", ascending=True)
        ax_a.barh([clean_axis_label(v, 42) for v in plot["group_key"]], plot["n_target_model_cases"])
        ax_a.set_xlabel("Number of target × model cases in top-risk decile")
    ax_a.set_title("A. Recurring hard-domain motifs")

    # Target by domain-dimension contribution matrix
    dim_rows = []
    for (target, dim), g in temp.groupby(["target", "group_dimension"]):
        dim_rows.append({
            "target": target,
            "group_dimension": dim,
            "mean_mae": g["mae_g"].mean(),
            "top_decile_fraction": float((g["risk_quantile"] >= 0.90).mean()),
        })
    dim_df = pd.DataFrame(dim_rows)
    save_figure_data(dim_df, out_base.name, "panel_b_domain_dimension_summary")
    if not dim_df.empty:
        mat = dim_df.pivot_table(index="target", columns="group_dimension", values="top_decile_fraction", aggfunc="mean")
        mat = mat.reindex(TARGET_COLUMNS).rename(index=short_target_label, columns=lambda x: clean_axis_label(x, 18))
        plot_matrix_heatmap(ax_b, mat, "B. Which domain definitions expose risk?", "Fraction in top-risk decile", fmt=".2f", annotate=True)

    for label, ax in zip(["A", "B"], axes):
        add_panel_label(ax, label)
    fig.suptitle("Recurring failure motifs across adsorption tasks", fontsize=16)
    safe_tight_layout(fig, rect=[0, 0, 1, 0.94])
    savefig(fig, out_base)


def create_composite_si_figures(agg: Dict[str, pd.DataFrame]) -> None:
    """Create compact SI composites rather than many one-panel plots."""
    for target_col in TARGET_COLUMNS:
        for model_name in ["hgb", "rf"]:
            out_base = DIRS["si_figures"] / f"Figure_SI_failure_atlas__{file_target_label(target_col)}__{model_name}"
            plot_primary_failure_atlas_composite(agg, target_col, model_name, out_base)
        out_base = DIRS["si_figures"] / f"Figure_SI_trust_disagreement__{file_target_label(target_col)}"
        plot_trust_and_disagreement_composite(agg, target_col, out_base)


def run_figure_step(step_name: str, func, *args, **kwargs) -> None:
    """Run one figure step without allowing a single figure to stop the pipeline."""
    try:
        log(f"Creating figure step: {step_name}")
        func(*args, **kwargs)
    except Exception as exc:
        err_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log(f"WARNING: figure step failed but pipeline will continue: {step_name} | {err_text}")


def create_all_figures(agg: Dict[str, pd.DataFrame]) -> None:
    """Create the revised story-oriented composite figure set.

    The main text is now anchored on CO2 at 0.15 bar because it gives the most
    chemically and process-relevant failure-map narrative: clear pore/density
    domains, clear local trust regions, and direct relevance to low-pressure CO2
    capture. Methane at 65 bar remains a valuable SI robustness case because it
    shows a stronger high-pressure storage-style failure mode and larger model
    disagreement in some high-PLD regions.
    """
    log("Creating revised composite manuscript and SI figures.")

    main_model = "hgb"

    run_figure_step(
        "Figure 1 workflow and target selection",
        plot_workflow_and_target_selection_composite,
        agg,
        DIRS["manuscript_figures"] / "Figure_1_workflow_and_target_selection",
    )
    run_figure_step(
        "Figure 2 benchmark reliability landscape",
        plot_benchmark_reliability_landscape_composite,
        agg,
        DIRS["manuscript_figures"] / "Figure_2_benchmark_reliability_landscape",
    )
    run_figure_step(
        "Figure 3 primary CO2 failure atlas",
        plot_primary_failure_atlas_composite,
        agg,
        target_col=PRIMARY_TARGET,
        model_name=main_model,
        out_base=DIRS["manuscript_figures"] / "Figure_3_primary_CO2_failure_atlas",
    )
    run_figure_step(
        "Figure 4 primary CO2 trust and disagreement",
        plot_trust_and_disagreement_composite,
        agg,
        target_col=PRIMARY_TARGET,
        out_base=DIRS["manuscript_figures"] / "Figure_4_primary_CO2_trust_and_disagreement",
    )
    run_figure_step(
        "Figure 5 recurring failure motifs",
        plot_recurring_failure_motifs_composite,
        agg,
        DIRS["manuscript_figures"] / "Figure_5_recurring_failure_motifs",
    )

    # Keep the statistical composite, but now for the CO2 main-text target.
    run_figure_step(
        "Figure 6 statistical reliability composite",
        plot_statistical_reliability_composite,
        metrics_df=agg["benchmark_split_metrics_df"] if "benchmark_split_metrics_df" in agg else pd.DataFrame(),
        novelty_df=agg["novelty_df"],
        dis_df=agg["target_to_disagreement"].get(PRIMARY_TARGET, pd.DataFrame()),
        target_col=PRIMARY_TARGET,
        out_base=DIRS["manuscript_figures"] / "Figure_6_statistical_reliability_composite",
    )

    # One-panel trace figures are retained for reproducibility and auditability.
    run_figure_step("SI original workflow trace", plot_workflow_figure, DIRS["si_figures"] / "Figure_S1_original_workflow_trace")
    run_figure_step(
        "SI original domain heatmap trace",
        plot_domain_resolved_heatmap,
        pred_df=agg["all_pred_df"],
        target_col=PRIMARY_TARGET,
        model_name=main_model,
        out_base=DIRS["si_figures"] / "Figure_S2_original_domain_heatmap_trace",
    )
    run_figure_step(
        "SI original novelty-error trace",
        plot_novelty_vs_error,
        pred_df=agg["all_pred_df"],
        target_col=PRIMARY_TARGET,
        model_name=main_model,
        out_base=DIRS["si_figures"] / "Figure_S3_original_novelty_error_trace",
    )

    # SI composite atlas for all targets and the two nonlinear models.
    run_figure_step("SI composite atlases", create_composite_si_figures, agg)




def export_curated_tables(agg: Dict[str, pd.DataFrame]) -> None:
    log("Exporting curated manuscript and SI tables.")

    coverage_df = agg["coverage_df"]
    benchmark_df = agg["benchmark_df"]
    hard_df = agg["hard_df"]
    disagreement_summary_df = agg["disagreement_summary_df"]

    # Table 1: merged group coverage statistics
    table1 = coverage_df.copy()
    safe_to_csv(table1, DIRS["manuscript_tables"] / "Table_1_coverage_statistics.csv", index=False)
    render_table_image(table1, "Table 1. Merged group coverage statistics", DIRS["tables_rendered"] / "Table_1_coverage_statistics.png")

    # Table 2: main benchmark and local reliability summary
    table2 = benchmark_df.copy()
    safe_to_csv(table2, DIRS["manuscript_tables"] / "Table_2_main_benchmark_summary.csv", index=False)
    render_table_image(table2, "Table 2. Main benchmark summary", DIRS["tables_rendered"] / "Table_2_main_benchmark_summary.png")

    # Table 3: hardest groups
    table3 = (
        hard_df[(hard_df["target"] == PRIMARY_TARGET) & (hard_df["model_name"] == "hgb")]
        .sort_values("composite_hardness_score", ascending=True)
        .head(40)
        .copy()
    )
    safe_to_csv(table3, DIRS["manuscript_tables"] / "Table_3_hardest_groups.csv", index=False)
    render_table_image(table3, "Table 3. Hardest groups by error, rank failure, and novelty", DIRS["tables_rendered"] / "Table_3_hardest_groups.png")

    # Table 4: agreement/disagreement summary
    table4 = disagreement_summary_df.copy()
    safe_to_csv(table4, DIRS["manuscript_tables"] / "Table_4_agreement_disagreement_summary.csv", index=False)
    render_table_image(table4, "Table 4. Agreement/disagreement summary", DIRS["tables_rendered"] / "Table_4_agreement_disagreement_summary.png")

    # SI tables
    for name, df in [
        ("Table_S1_full_coverage_statistics.csv", coverage_df),
        ("Table_S2_full_benchmark_summary.csv", benchmark_df),
        ("Table_S3_full_hard_domain_rankings.csv", hard_df),
        ("Table_S4_full_agreement_disagreement_summary.csv", disagreement_summary_df),
        ("Table_S5_novelty_error_relationships.csv", agg["novelty_df"]),
        ("Table_S6_group_level_error_summary.csv", agg["group_df"]),
    ]:
        safe_to_csv(df, DIRS["si_tables"] / name, index=False)


# =============================================================================
# Main execution
# =============================================================================

def save_run_configuration() -> None:
    config = {
        "project_name": PROJECT_NAME,
        "output_root": str(OUTPUT_ROOT),
        "targets": TARGET_COLUMNS,
        "primary_target": PRIMARY_TARGET,
        "si_secondary_target": SI_SECONDARY_TARGET,
        "n_outer_splits": N_OUTER_SPLITS,
        "test_size": TEST_SIZE,
        "base_random_seed": BASE_RANDOM_SEED,
        "top_fraction": TOP_FRACTION,
        "min_group_size": MIN_GROUP_SIZE,
        "model_specs": list(MODEL_SPECS.keys()),
        "model_panel_note": "Lightweight publication-reasonable defaults: Ridge baseline; RF with 80 controlled-depth trees and row/feature subsampling; HGB with 120 iterations, shallow trees, and early stopping.",
        "model_parameters": {name: model.get_params() for name, model in MODEL_SPECS.items()},
        "n_jobs_default_user_value": N_JOBS,
        "n_jobs_random_forest_runtime": MODEL_SPECS["rf"].get_params().get("n_jobs"),
    }
    save_json(config, DIRS["exports"] / "run_configuration.json")


def export_summary_readme(agg: Dict[str, pd.DataFrame]) -> None:
    text = f"""
    Interpretable Failure Maps pipeline completed.

    Main folders
    ------------
    logs/
        Runtime logs.

    data_processed/
        Merged master table and processed data snapshots.

    split_definitions/
        Persistent split definitions by target.

    results/models/
        Saved fitted models by target/model/split.

    results/predictions/
        Per-sample prediction outputs. These are the key reusable files for
        downstream analysis without re-fitting.

    results/metrics/
        Split-wise benchmark metrics.

    results/tables/
        Master tables used for manuscript and SI assembly.

    manuscript_assets/figures/
        Main-text figure files (PNG and PDF).

    supplementary_assets/figures/
        SI figure files.

    manuscript_assets/tables/
        Curated main-text tables.

    supplementary_assets/tables/
        Curated SI tables.

    Notes
    -----
    * The main manuscript figure set uses:
          PRIMARY_TARGET = {PRIMARY_TARGET}
          MAIN_MODEL = hgb
      Methane at 65 bar is retained as a robustness-focused SI case rather than the main anchor.
    * Additional targets and model variants were saved for SI and robustness.
    * The ML panel uses lightweight publication-reasonable defaults to reduce runtime while preserving Ridge/RF/HGB comparisons.
    * All prediction-level outputs were retained so the paper can be extended
      later without repeating model fitting.
    """
    path = DIRS["exports"] / "README_outputs.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(text).strip() + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    resolved_n_jobs = configure_runtime(args)

    log("=" * 80)
    log("Starting Interpretable Failure Maps pipeline.")
    log(f"Script directory: {SCRIPT_DIR}")
    log(f"Output root: {OUTPUT_ROOT}")
    log(f"Requested n_jobs: {args.n_jobs}; resolved RandomForest n_jobs: {resolved_n_jobs}")
    log("Lightweight model defaults active: RF=80 trees/depth14/max_samples0.70; HGB=120 iterations/depth6/early stopping.")

    save_run_configuration()

    # 1) Load and merge master data
    df_master = load_and_merge_data(force_rerun=FORCE_RERUN_MERGE)

    # 2) Run all target/model/split fits
    metrics_df = run_model_panel(df_master)

    # 3) Aggregate all saved outputs
    agg = save_all_aggregated_outputs(df_master, metrics_df)

    # 4) Make tables and figures
    export_curated_tables(agg)

    if not getattr(args, "skip_figures", False):
        create_all_figures(agg)
    else:
        log("Skipping figure generation because --skip_figures was requested.")

    # 5) Summary notes
    export_summary_readme(agg)

    log("Pipeline completed successfully.")
    log("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Run interrupted by user (Ctrl+C). Saved checkpoints and completed outputs can be reused on the next run.")
        raise
    except Exception as exc:
        err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log("Fatal error encountered.\n" + err_text)
        raise

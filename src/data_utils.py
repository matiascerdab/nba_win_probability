# src/data_utils.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_pbp_features.parquet"

# === Definición oficial de columnas del modelo actual ===
BASE_STATE_COLS = [
    "score_diff",
    "seconds_remaining_game",
    "period",
]

PRE_GAME_COLS = [
    "h_ts_pct_before_w",
    "h_pts_prev_avg_w",
    "h_avg_games_played_ratio_before",
    "a_ts_pct_before_w",
    "a_pts_prev_avg_w",
    "a_avg_games_played_ratio_before",
]

IN_GAME_COLS = [
    "h_ts_pct_so_far",
    "h_efg_pct_so_far",
    "h_pct_fga_3pt_so_far",
    "a_ts_pct_so_far",
    "a_efg_pct_so_far",
    "a_pct_fga_3pt_so_far",
    "h_pts_so_far",
    "a_pts_so_far",
]

FEATURE_COLS = BASE_STATE_COLS + PRE_GAME_COLS + IN_GAME_COLS
TARGET_COL = "home_win"
# ========================================================


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}")
    return df


def get_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limpia el dataset y devuelve X, y listos para scikit-learn:

    - Filtra filas donde home_win es NaN.
    - Quita inf y NaN en las features (los rellena con 0.0).
    - Devuelve y como entero 0/1.
    """
    df = df.copy()

    # 1) Filtrar target no nulo
    before = len(df)
    df = df[df[TARGET_COL].notna()]
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[data_utils] Filtradas {dropped} filas con {TARGET_COL} NaN (de {before}).")

    # 2) Asegurar que todas las columnas de features existen
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas en el dataset: {missing_cols}")

    # 3) Limpiar X: inf -> NaN -> 0.0
    X = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    # 4) y como entero
    y = df[TARGET_COL].astype(int).to_numpy()

    return X, y

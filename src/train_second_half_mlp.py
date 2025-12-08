# src/train_second_half_mlp.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_second_half_dataset.parquet"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "second_half_mlp.pkl"
SCALER_PATH = MODELS_DIR / "second_half_scaler.pkl"


FEATURE_COLS = [
    "margin_HT",
    #"pts_diff_HT",
    #"ts_diff_HT",
    #"ts_diff_pre",
    #"pts_prev_diff_pre",
    #"pregame_spread_home",
    "h2_home_spread",
]

TARGET_COL_RESID = "resid_2H"
TARGET_COL_MARGIN = "margin_2H_real"


def load_second_half_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de 2º tiempo en:\n  {path}\n"
            "Primero corre build_second_half_dataset.py."
        )
    print(f"[train_second_half_mlp] Cargando dataset desde {path}")
    df = pd.read_parquet(path)
    return df


def train_mlp(df: pd.DataFrame) -> None:
    # Filtramos filas con features completas
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas de features en el dataset: {missing_cols}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL_RESID, TARGET_COL_MARGIN, "h2_home_spread"]).copy()

    # DEBUG: ver si hay partidos duplicados
    dupes = df[["season", "game_date", "home_team", "away_team"]].duplicated().sum()
    print("Duplicados exactos de partido:", dupes)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_resid = df[TARGET_COL_RESID].to_numpy(dtype=float)
    y_margin = df[TARGET_COL_MARGIN].to_numpy(dtype=float)
    spread_2H = df["h2_home_spread"].to_numpy(dtype=float)

    X_train, X_test, y_resid_train, y_resid_test, y_margin_train, y_margin_test, spread_train, spread_test = \
        train_test_split(
            X,
            y_resid,
            y_margin,
            spread_2H,
            test_size=0.2,
            random_state=42,
        )

    # Escalado estándar de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP "deep-ish": dos capas ocultas ReLU
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        verbose=True,
    )

    print("[train_second_half_mlp] Entrenando MLP sobre el residuo 2H...")
    mlp.fit(X_train_scaled, y_resid_train)

    # --- Evaluación: baseline mercado vs modelo ---

    # Baseline mercado: predicción = línea de 2H
    y_pred_market_train = spread_train
    y_pred_market_test = spread_test

    # Modelo: linea + residuo
    y_resid_hat_train = mlp.predict(X_train_scaled)
    y_resid_hat_test = mlp.predict(X_test_scaled)

    y_pred_model_train = spread_train + y_resid_hat_train
    y_pred_model_test = spread_test + y_resid_hat_test

    # Métricas (RMSE y MAE) sobre el margen real en 2H
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n[train_second_half_mlp] === RESULTADOS TRAIN ===")
    print(f"RMSE mercado: {rmse(y_margin_train, y_pred_market_train):.4f}")
    print(f"RMSE modelo : {rmse(y_margin_train, y_pred_model_train):.4f}")
    print(f"MAE  mercado: {mean_absolute_error(y_margin_train, y_pred_market_train):.4f}")
    print(f"MAE  modelo : {mean_absolute_error(y_margin_train, y_pred_model_train):.4f}")

    print("\n[train_second_half_mlp] === RESULTADOS TEST ===")
    print(f"RMSE mercado: {rmse(y_margin_test, y_pred_market_test):.4f}")
    print(f"RMSE modelo : {rmse(y_margin_test, y_pred_model_test):.4f}")
    print(f"MAE  mercado: {mean_absolute_error(y_margin_test, y_pred_market_test):.4f}")
    print(f"MAE  modelo : {mean_absolute_error(y_margin_test, y_pred_model_test):.4f}")

    # Guardamos modelo + scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[train_second_half_mlp] Modelo guardado en {MODEL_PATH}")
    print(f"[train_second_half_mlp] Scaler guardado en {SCALER_PATH}")

    # Métrica extra: acierto de lado de la línea (quién cubre el spread 2H)
    def sign(x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    # Diferencia real vs línea (2H)
    D_real_train = y_margin_train - spread_train
    D_real_test = y_margin_test - spread_test

    D_model_train = y_pred_model_train - spread_train
    D_model_test = y_pred_model_test - spread_test

    hit_train = (sign(D_real_train) == sign(D_model_train)).mean()
    hit_test = (sign(D_real_test) == sign(D_model_test)).mean()

    print(f"\n[train_second_half_mlp] === HIT RATE LADO DE LÍNEA ===")
    print(f"Train hit rate: {hit_train:.3f}")
    print(f"Test  hit rate: {hit_test:.3f}")

        # === Análisis del lado real del spread (quién cubre 2H) ===
    # cover_home = 1 si el home está "por encima" de la línea 2H, 0 si no
    cover_home_train = (D_real_train > 0).astype(int)
    cover_home_test  = (D_real_test  > 0).astype(int)

    prop_home_covers_train = cover_home_train.mean()
    prop_home_covers_test  = cover_home_test.mean()

    print("\n[train_second_half_mlp] === DISTRIBUCIÓN COVER 2H (REAL) ===")
    print(f"Proporción home cubre (train): {prop_home_covers_train:.3f}")
    print(f"Proporción home cubre (test) : {prop_home_covers_test:.3f}")

def main() -> None:
    df = load_second_half_data()
    train_mlp(df)


if __name__ == "__main__":
    main()

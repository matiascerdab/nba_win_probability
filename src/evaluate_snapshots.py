import pandas as pd

# src/evaluate_snapshots.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)

from data_utils import load_dataset, FEATURE_COLS, TARGET_COL

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_pbp_features.parquet"
MODEL_PATH = ROOT / "models" / "logreg_winprob.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()

    # Misma limpieza que en data_utils.get_X_y:
    # inf -> NaN -> 0.0
    X = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    proba = model.predict_proba(X)[:, 1]
    df["pred_win_home"] = proba
    return df


def snapshot_masks(df: pd.DataFrame) -> dict:
    """
    Define las filas (mask) para cada snapshot:

    - start_game : primer estado de cada partido
    - start_Q2   : primer estado del 2º cuarto
    - end_Q1     : último estado del 1er cuarto
    - end_Q2     : último estado del 2º cuarto
    - end_Q3     : último estado del 3er cuarto
    """

    masks: dict[str, np.ndarray] = {}

    # Orden temporal razonable
    df_sorted = df.sort_values(["game_id", "seconds_elapsed_game"]).copy()

    # 1) Inicio del partido: primer estado por game
    idx_start = (
        df_sorted.groupby("game_id", sort=False)
        .head(1)
        .index
    )
    masks["start_game"] = df.index.isin(idx_start)

    # 1b) Inicio del segundo cuarto: primer estado de period == 2
    def first_state_of_period(p: int):
        sub = df_sorted[df_sorted["period"] == p]
        if sub.empty:
            # índice vacío
            return df.index[df.index == -1]
        return (
            sub.groupby("game_id", sort=False)
            .head(1)
            .index
        )

    idx_start_q2 = first_state_of_period(2)
    masks["start_Q2"] = df.index.isin(idx_start_q2)

    # Helper para fin de cada cuarto
    def last_state_of_period(p: int):
        sub = df_sorted[df_sorted["period"] == p]
        if sub.empty:
            # índice vacío
            return df.index[df.index == -1]
        return (
            sub.groupby("game_id", sort=False)
            .tail(1)
            .index
        )

    idx_q1 = last_state_of_period(1)
    idx_q2 = last_state_of_period(2)
    idx_q3 = last_state_of_period(3)

    masks["end_Q1"] = df.index.isin(idx_q1)
    masks["end_Q2"] = df.index.isin(idx_q2)
    masks["end_Q3"] = df.index.isin(idx_q3)

    return masks



def compute_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    print("Cargando dataset y modelo...")
    df = load_dataset(DATA_PATH)

    # Filtramos filas sin target y lo convertimos a int (0/1)
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    model = load_model()
    df = add_predictions(df, model)

    masks = snapshot_masks(df)

    print("\nMétricas por snapshot:")
    for name, m in masks.items():
        df_s = df[m]
        if df_s.empty:
            print(f"\n{name}: SIN FILAS (revisa las condiciones de tiempo)")
            continue

        metrics = compute_metrics(
            df_s[TARGET_COL].to_numpy(),
            df_s["pred_win_home"].to_numpy(),
        )
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k == "n":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

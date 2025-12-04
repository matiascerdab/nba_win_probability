# src/build_model_vs_bookmaker_dataset.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from data_utils import load_dataset, TARGET_COL
from evaluate_snapshots import load_model, add_predictions, snapshot_masks

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_pbp_features.parquet"
OUT_PATH = ROOT / "data" / "processed" / "nba_model_vs_bookmaker_snapshots.parquet"


def build_model_vs_bookmaker_dataset(
    data_path: Path | str = DATA_PATH,
    out_path: Path | str = OUT_PATH,
) -> pd.DataFrame:
    """
    Construye un dataset a nivel (partido, snapshot) con:

    - Probabilidad de victoria del modelo (p_home_model).
    - Odds y probabilidades implícitas del bookmaker (p_home_book, etc.).
    - Dos snapshots por partido:
        * pre_game  -> primer estado del partido.
        * start_Q2  -> primer estado del 2º cuarto.

    Este dataset NO se usa para entrenar el modelo de win probability;
    está pensado para que otra red lo consuma y busque oportunidades de apuesta.
    """
    data_path = Path(data_path)
    out_path = Path(out_path)

    print(f"[build_model_vs_bookmaker_dataset] Cargando dataset base desde {data_path}")
    df = load_dataset(data_path)

    # Filtramos filas con target válido y lo pasamos a int
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Cargamos el modelo actual (logístico ahora, deep learning en el futuro)
    print("[build_model_vs_bookmaker_dataset] Cargando modelo de win probability...")
    model = load_model()

    # Añadimos predicciones del modelo: pred_win_home
    print("[build_model_vs_bookmaker_dataset] Calculando probabilidades del modelo...")
    df = add_predictions(df, model)
    df["p_home_model"] = df["pred_win_home"].astype(float)

    # Definimos máscaras de snapshots (start_game, start_Q2, end_Q1/2/3, ...)
    masks = snapshot_masks(df)

    # Nos quedamos solo con los snapshots que te interesan ahora:
    snapshot_map = {
        "start_game": "pre_game",
        "start_Q2": "start_Q2",
    }

    frames: list[pd.DataFrame] = []

    for internal_name, public_name in snapshot_map.items():
        if internal_name not in masks:
            print(
                f"[build_model_vs_bookmaker_dataset] WARNING: snapshot '{internal_name}' "
                f"no está definido en snapshot_masks. Se ignora."
            )
            continue

        m = masks[internal_name]
        df_s = df[m].copy()
        df_s["snapshot"] = public_name
        frames.append(df_s)

    if not frames:
        raise RuntimeError(
            "[build_model_vs_bookmaker_dataset] No se generó ningún snapshot. "
            "Revisa snapshot_masks en evaluate_snapshots.py."
        )

    df_snapshots = pd.concat(frames, ignore_index=True)

    # Columnas mínimas que esperamos tener en el dataset base
    required_cols = [
        "season",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        TARGET_COL,          # home_win
        "home_ml",
        "away_ml",
        "p_home_book",
        "p_away_book",
        "p_home_model",
        "snapshot",
    ]

    missing = [c for c in required_cols if c not in df_snapshots.columns]
    if missing:
        raise KeyError(
            "[build_model_vs_bookmaker_dataset] Faltan columnas en el dataset base: "
            + ", ".join(missing)
        )

    df_final = df_snapshots[required_cols].copy()

    # Aseguramos tipos razonables
    df_final["season"] = df_final["season"].astype(int)
    df_final["home_win"] = df_final[TARGET_COL].astype(int)
    df_final["game_date"] = pd.to_datetime(df_final["game_date"])

    # Orden opcional de filas: por season, game_date, game_id, snapshot
    df_final = df_final.sort_values(
        ["season", "game_date", "game_id", "snapshot"]
    ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build_model_vs_bookmaker_dataset] Guardando dataset final en {out_path}")
    df_final.to_parquet(out_path, index=False)
    print("[build_model_vs_bookmaker_dataset] Listo ✅")

    return df_final


def main() -> None:
    build_model_vs_bookmaker_dataset()


if __name__ == "__main__":
    main()

# src/build_second_half_dataset.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


# -----------------------------
# 1. Cargar estados PBP
# -----------------------------
def load_states() -> pd.DataFrame:
    """
    Carga el dataset de estados in-game (nba_pbp_features.parquet)
    generado por build_dataset.py.
    """
    path = PROCESSED_DIR / "nba_pbp_features.parquet"
    print(f"[build_second_half_dataset] Cargando estados desde: {path}")
    df = pd.read_parquet(path)

    df["game_id"] = df["game_id"].astype(str)
    df["season"] = df["season"].astype(int)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    return df


# -----------------------------
# 2. Estados al descanso (half-time)
# -----------------------------
def get_halftime_states(states: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene el marcador al descanso.

    Si existiera 'desc', podríamos usar 'End of 2nd Period', pero en tu
    nba_pbp_features.parquet no está, así que:

      - period == 2
      - seconds_remaining_period == 0

    Luego, forzamos UNA fila por (season, game_id).
    """
    if "seconds_remaining_period" not in states.columns:
        raise KeyError(
            "El DataFrame de estados no tiene 'seconds_remaining_period'. "
            "No puedo localizar el half-time."
        )

    mask_ht = (states["period"] == 2) & (states["seconds_remaining_period"] == 0)
    print("[get_halftime_states] Usando periodo==2 & seconds_remaining_period==0 como half-time.")

    cols = ["season", "game_id"]
    if "game_date" in states.columns:
        cols.append("game_date")
    if "home_team" in states.columns:
        cols.append("home_team")
    if "away_team" in states.columns:
        cols.append("away_team")

    cols += ["h_pts_so_far", "a_pts_so_far"]

    for c in [
        "h_ts_pct_so_far", "a_ts_pct_so_far",
        "h_ts_pct_before_w", "a_ts_pct_before_w",
        "h_pts_prev_avg_w", "a_pts_prev_avg_w",
    ]:
        if c in states.columns:
            cols.append(c)

    ht = states.loc[mask_ht, cols].copy()

    ht["h_pts_HT"] = ht["h_pts_so_far"]
    ht["a_pts_HT"] = ht["a_pts_so_far"]
    ht["margin_HT"] = ht["h_pts_HT"] - ht["a_pts_HT"]
    ht["pts_diff_HT"] = ht["margin_HT"]

    if "h_ts_pct_so_far" in ht.columns and "a_ts_pct_so_far" in ht.columns:
        ht["h_ts_pct_HT"] = ht["h_ts_pct_so_far"]
        ht["a_ts_pct_HT"] = ht["a_ts_pct_so_far"]
        ht["ts_diff_HT"] = ht["h_ts_pct_HT"] - ht["a_ts_pct_HT"]

    if "h_ts_pct_before_w" in ht.columns and "a_ts_pct_before_w" in ht.columns:
        ht["ts_diff_pre"] = ht["h_ts_pct_before_w"] - ht["a_ts_pct_before_w"]

    if "h_pts_prev_avg_w" in ht.columns and "a_pts_prev_avg_w" in ht.columns:
        ht["pts_prev_diff_pre"] = ht["h_pts_prev_avg_w"] - ht["a_pts_prev_avg_w"]

    before = len(ht)
    ht = ht.sort_values(["season", "game_id"]).drop_duplicates(
        subset=["season", "game_id"], keep="last"
    )
    after = len(ht)
    dropped = before - after

    print(
        f"[get_halftime_states] Encontrados {after} registros de halftime "
        f"(se eliminaron {dropped} duplicados por (season, game_id))."
    )
    return ht


# -----------------------------
# 3. Margen final del partido
# -----------------------------
def get_final_margin(states: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene el marcador final del partido usando la última fila
    por (season, game_id) según el orden temporal.
    """
    if "seconds_remaining_period" not in states.columns:
        raise KeyError(
            "El DataFrame de estados no tiene 'seconds_remaining_period'. "
            "No puedo localizar el final del partido."
        )

    df_final = (
        states
        .sort_values(["season", "game_id", "period", "seconds_remaining_period"])
        .groupby(["season", "game_id"])
        .tail(1)
        .copy()
    )
    print("[get_final_margin] Usando última fila de cada partido como final.")

    df_final["margin_final"] = df_final["h_pts_so_far"] - df_final["a_pts_so_far"]

    cols_final = ["season", "game_id", "margin_final"]
    if "game_date" in df_final.columns:
        cols_final.append("game_date")
    if "home_team" in df_final.columns:
        cols_final.append("home_team")
    if "away_team" in df_final.columns:
        cols_final.append("away_team")

    final = df_final[cols_final].copy()

    before = len(final)
    final = final.sort_values(["season", "game_id"]).drop_duplicates(
        subset=["season", "game_id"], keep="last"
    )
    after = len(final)
    dropped = before - after

    print(
        f"[get_final_margin] Encontrados {after} registros de final de partido "
        f"(se eliminaron {dropped} duplicados por (season, game_id))."
    )
    return final


# -----------------------------
# 4. PUNTOS POR CUARTO
# -----------------------------
def compute_quarter_points(states: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula puntos de cada equipo por cuarto (Q1..Q4), a partir de h_pts_so_far/a_pts_so_far.

    Estrategia:
      - Para periodos 1..4, tomamos la última fila de cada (season, game_id, period)
        (ordenada por seconds_remaining_period).
      - Eso nos da el marcador ACUMULADO al final de cada cuarto.
      - A partir de eso, calculamos los puntos de cada cuarto por diferencia
        de acumulados.
    """
    if "seconds_remaining_period" not in states.columns:
        raise KeyError(
            "El DataFrame de estados no tiene 'seconds_remaining_period'; "
            "no puedo calcular puntos por cuarto."
        )

    # Nos quedamos solo con los 4 primeros periodos (no OTs) para los cuartos.
    tmp = (
        states[states["period"].between(1, 4)]
        .sort_values(["season", "game_id", "period", "seconds_remaining_period"])
        .groupby(["season", "game_id", "period"])
        .tail(1)
        .copy()
    )

    # Pivot a formato ancho: columnas h_pts_so_far/a_pts_so_far por periodo
    wide = tmp.pivot(
        index=["season", "game_id"],
        columns="period",
        values=["h_pts_so_far", "a_pts_so_far"],
    )

    # Renombrar columnas: h_end_q1, h_end_q2, ...
    wide.columns = [
        f"{side}_end_q{per}"
        for (side, per) in wide.columns
    ]
    wide = wide.reset_index()

    # Aseguramos que las columnas existan; si no, las rellenamos con NaN
    for q in [1, 2, 3, 4]:
        for side in ["h", "a"]:
            col = f"{side}_end_q{q}"
            if col not in wide.columns:
                wide[col] = np.nan

    # Puntos por cuarto = diferencia de acumulados
    wide["h_q1_pts"] = wide["h_end_q1"]
    wide["a_q1_pts"] = wide["a_end_q1"]

    wide["h_q2_pts"] = wide["h_end_q2"] - wide["h_end_q1"]
    wide["a_q2_pts"] = wide["a_end_q2"] - wide["a_end_q1"]

    wide["h_q3_pts"] = wide["h_end_q3"] - wide["h_end_q2"]
    wide["a_q3_pts"] = wide["a_end_q3"] - wide["a_end_q2"]

    wide["h_q4_pts"] = wide["h_end_q4"] - wide["h_end_q3"]
    wide["a_q4_pts"] = wide["a_end_q4"] - wide["a_end_q3"]

    # También calculamos un margin_2H basado en Q3+Q4 (solo para debug)
    wide["margin_2H_q34"] = (wide["h_q3_pts"] + wide["h_q4_pts"]) - (
        wide["a_q3_pts"] + wide["a_q4_pts"]
    )

    print(
        f"[compute_quarter_points] Generados puntos por cuarto para {len(wide)} partidos "
        "(Q1..Q4, más margin_2H_q34)."
    )

    return wide


# -----------------------------
# 5. Cargar betting_history.csv
# -----------------------------
def load_second_half_spreads() -> pd.DataFrame:
    """
    Carga betting_history.csv (exportado desde la base SQLite de
    https://www.kaggle.com/datasets/visualize25/basketball-betting-dataset)

    Columnas que usamos:
      - GAME_ID            -> game_id
      - Date               -> game_date
      - HomeSpread_AtOpen  -> pregame_spread_home
      - 2H_HomeSpread      -> h2_home_spread
    """
    path = EXTERNAL_DIR / "basketball_betting" / "betting_history.csv"
    print(f"[build_second_half_dataset] Cargando 2H spreads desde: {path}")

    odds = pd.read_csv(path)

    rename_map = {
        "GAME_ID": "game_id",
        "Date": "game_date",
        "HomeSpread_AtOpen": "pregame_spread_home",
        "2H_HomeSpread": "h2_home_spread",
    }
    odds = odds.rename(columns=rename_map)

    odds["game_id"] = odds["game_id"].astype(str)
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.normalize()

    keep_cols = ["game_id", "game_date", "pregame_spread_home", "h2_home_spread"]
    odds = odds[keep_cols].drop_duplicates(subset=["game_id"])

    print(
        "[build_second_half_dataset] Ejemplo spreads_2h "
        "(game_id, game_date, pregame_spread_home, h2_home_spread):"
    )
    print(odds.head(10))

    return odds


# -----------------------------
# 6. Construir dataset 2H
# -----------------------------
def build_second_half_dataset() -> pd.DataFrame:
    states = load_states()

    # 1) Half-time + margen final
    ht = get_halftime_states(states)
    final = get_final_margin(states)

    base_2h = ht.merge(
        final[["season", "game_id", "margin_final"]],
        on=["season", "game_id"],
        how="inner",
        validate="1:1",
    )

    # margen real del segundo tiempo (Q3+Q4+OTs)
    base_2h["margin_2H_real"] = base_2h["margin_final"] - base_2h["margin_HT"]
    base_2h["game_id"] = base_2h["game_id"].astype(str)

    # 2) PUNTOS POR CUARTO
    quarters = compute_quarter_points(states)
    base_2h = base_2h.merge(
        quarters,
        on=["season", "game_id"],
        how="left",
        validate="1:1",
    )

    # 3) Spreads 2H desde betting_history.csv
    spreads_2h = load_second_half_spreads()

    print("\n=== [INSPECCIÓN DE BASES ANTES DEL MERGE DE SPREADS] ===\n")
    print(f"base_2h.shape: {base_2h.shape}")
    print(f"spreads_2h.shape: {spreads_2h.shape}")

    print("\nColumnas base_2h (primeras):")
    print(list(base_2h.columns)[:25])

    print("\nColumnas spreads_2h:")
    print(list(spreads_2h.columns))

    print("\nEjemplo base_2h (season, game_id):")
    print(base_2h[["season", "game_id"]].head(10))

    print(
        "\nEjemplo spreads_2h (game_id, game_date, pregame_spread_home, h2_home_spread):"
    )
    print(
        spreads_2h[
            ["game_id", "game_date", "pregame_spread_home", "h2_home_spread"]
        ].head(10)
    )

    # 4) Merge SOLO por game_id con spreads
    df = base_2h.merge(
        spreads_2h,
        on="game_id",
        how="inner",
        validate="1:1",
    )

    # 5) Residuo del mercado en 2H (con margin_2H_real "clásico")
    df["resid_2H"] = df["margin_2H_real"] - df["h2_home_spread"]

    print("\n[build_second_half_dataset] Ejemplo de columnas del dataset final:")
    print(df.head(3).T)

    return df


# -----------------------------
# 7. main()
# -----------------------------
def main():
    df = build_second_half_dataset()

    out_path = PROCESSED_DIR / "nba_second_half_dataset.parquet"
    print(f"[build_second_half_dataset] Guardando dataset 2H en: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("[build_second_half_dataset] Listo ✅")


if __name__ == "__main__":
    main()

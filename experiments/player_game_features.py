import pandas as pd
import numpy as np


def build_player_game_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de games_df (jugador-partido con stats del partido),
    construye features jugador-partido usando SOLO la info de partidos anteriores.

    Espera games_df con columnas:
    ['season', 'game_id', 'team_id', 'player_id',
     'pts', 'reb', 'stl', 'blk', 'tov', 'pf',
     'fga', 'fgm', 'tpa', 'tpm', 'fta', 'ftm']

    Devuelve un DataFrame con una fila por jugador-partido y columnas extra:
    - games_played_before
    - cum_*_before (acumulados previos)
    - *_prev_avg (promedios previos)
    - fg_pct_before, tp_pct_before, ft_pct_before, ts_pct_before
    - team_games_before
    - games_played_ratio_before
    """

    df = games_df.copy()

    required_cols = [
        "season", "game_id", "team_id", "player_id",
        "pts", "reb", "stl", "blk", "tov", "pf",
        "fga", "fgm", "tpa", "tpm", "fta", "ftm",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in games_df: {missing}")

    # Ordenamos para que los cumulativos sigan el orden cronológico de los partidos
    df = df.sort_values(["season", "team_id", "player_id", "game_id"]).reset_index(drop=True)

    stat_cols = ["pts", "reb", "stl", "blk", "tov", "pf",
                 "fga", "fgm", "tpa", "tpm", "fta", "ftm"]

    # ============================
    # 1) Historial por jugador (season+team+player)
    # ============================
    g_player = df.groupby(["season", "team_id", "player_id"], sort=False)

    # cuántos partidos llevaba antes (0 para el primer game de ese jugador)
    df["games_played_before"] = g_player.cumcount()

    # acumulados previos de cada stat (sin incluir el game actual)
    for col in stat_cols:
        cum_col = g_player[col].cumsum()
        df[f"cum_{col}_before"] = cum_col - df[col]

    # medias previas por partido
    g_before = df["games_played_before"].replace(0, np.nan)

    df["pts_prev_avg"] = df["cum_pts_before"] / g_before
    df["reb_prev_avg"] = df["cum_reb_before"] / g_before
    df["stl_prev_avg"] = df["cum_stl_before"] / g_before
    df["blk_prev_avg"] = df["cum_blk_before"] / g_before
    df["tov_prev_avg"] = df["cum_tov_before"] / g_before

    # rellenamos NaN de los primeros partidos con 0
    prev_avg_cols = [
        "pts_prev_avg", "reb_prev_avg",
        "stl_prev_avg", "blk_prev_avg", "tov_prev_avg",
    ]
    df[prev_avg_cols] = df[prev_avg_cols].fillna(0.0)

    # ============================
    # 2) Porcentajes y TS% previos
    # ============================
    # FG%, 3P%, FT% previos (sobre stats acumuladas antes)
    df["fg_pct_before"] = np.where(
        df["cum_fga_before"] > 0,
        df["cum_fgm_before"] / df["cum_fga_before"],
        0.0,
    )

    df["tp_pct_before"] = np.where(
        df["cum_tpa_before"] > 0,
        df["cum_tpm_before"] / df["cum_tpa_before"],
        0.0,
    )

    df["ft_pct_before"] = np.where(
        df["cum_fta_before"] > 0,
        df["cum_ftm_before"] / df["cum_fta_before"],
        0.0,
    )

    # TS% previo: usa pts acumulados antes y volumen de tiros acumulado antes
    denom_ts = 2 * (df["cum_fga_before"] + 0.44 * df["cum_fta_before"])
    df["ts_pct_before"] = np.where(
        denom_ts > 0,
        df["cum_pts_before"] / denom_ts,
        0.0,
    )

    # ============================
    # 3) Historial del equipo
    # ============================
    # team_games_before: número de partidos que lleva el equipo antes de este game
    team_games = (
        df[["season", "team_id", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "team_id", "game_id"])
    )
    team_games["team_games_before"] = team_games.groupby(
        ["season", "team_id"], sort=False
    ).cumcount()

    df = df.merge(
        team_games,
        on=["season", "team_id", "game_id"],
        how="left",
        validate="many_to_many",
    )

    # ratio: partidos jugados por el jugador / partidos jugados por el equipo
    df["games_played_ratio_before"] = np.where(
        df["team_games_before"] > 0,
        df["games_played_before"] / df["team_games_before"],
        0.0,
    )

    return df

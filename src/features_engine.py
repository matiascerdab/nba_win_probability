# src/features_engine.py

"""
Módulo central de ingeniería de features.

Contiene toda la lógica para pasar de los CSV de play-by-play de una temporada
a un dataset de estados in-game con:

- base_state_cols  (score_diff, seconds_remaining_game, period, etc.)
- pre_game_cols    (h_ts_pct_before_w, a_pts_prev_avg_w, ...)
- in_game_cols     (h_ts_pct_so_far, h_pts_so_far, ...)

Este módulo reemplaza:
    - games_df.py
    - player_game_features.py
    - team_game_features.py
    - in_game_team_states.py
    - build_season_states.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------------
# 1) games_df: jugador-partido con stats del partido
# --------------------------------------------------------

def build_games_df_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()

    df = df[(df["playerid"].notna()) & (df["playerid"] != 0)].copy()

    for col in ["type", "subtype", "desc", "player"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    shot_mask = df["type"].isin(["Made Shot", "Missed Shot"])
    is_made = df["type"] == "Made Shot"
    is_3pt = shot_mask & df["desc"].str.contains("3PT", na=False)

    df["fga"] = shot_mask.astype(int)
    df["fgm"] = is_made.astype(int)
    df["tpa"] = is_3pt.astype(int)
    df["tpm"] = (is_made & is_3pt).astype(int)

    ft_mask = df["type"] == "Free Throw"
    df["fta"] = ft_mask.astype(int)
    df["ftm"] = (ft_mask & ~df["desc"].str.startswith("MISS")).astype(int)

    df["reb"] = (df["type"] == "Rebound").astype(int)
    df["stl"] = df["desc"].str.contains("STEAL", na=False).astype(int)
    df["blk"] = df["desc"].str.contains("BLOCK", na=False).astype(int)
    df["tov"] = (df["type"] == "Turnover").astype(int)
    df["pf"] = (df["type"] == "Foul").astype(int)

    df["fg2m"] = df["fgm"] - df["tpm"]
    df["pts"] = 2 * df["fg2m"] + 3 * df["tpm"] + df["ftm"]

    agg_cols = [
        "pts", "reb", "stl", "blk", "tov", "pf",
        "fga", "fgm", "tpa", "tpm", "fta", "ftm",
    ]

    games_df = (
        df.groupby(["season", "gameid", "team", "playerid", "player"], as_index=False)[agg_cols]
        .sum()
    )

    games_df = games_df.rename(
        columns={
            "gameid": "game_id",
            "team": "team_id",
            "playerid": "player_id",
            "player": "player_name",
        }
    )

    return games_df


# --------------------------------------------------------
# 2) player_game_features: historial del jugador antes del partido
# --------------------------------------------------------

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


# --------------------------------------------------------
# 3) team_game_features: agregados por equipo-partido (pre-game)
# --------------------------------------------------------

def build_team_game_features(
    player_game_features_df: pd.DataFrame,
    core_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Agrega features jugador-partido a nivel equipo-partido (pre-game).

    Cada fila del resultado = (season, game_id, team_id).

    Usa SOLO columnas '..._before' (histórico antes de ese partido), por lo que
    no hay leakage.

    Espera en player_game_features_df al menos:
    - 'season', 'game_id', 'team_id', 'player_id', 'player_name'
    - 'games_played_before', 'team_games_before', 'games_played_ratio_before'
    - 'pts_prev_avg', 'reb_prev_avg', 'stl_prev_avg', 'blk_prev_avg', 'tov_prev_avg'
    - 'fg_pct_before', 'tp_pct_before', 'ft_pct_before', 'ts_pct_before'
    """

    df = player_game_features_df.copy()

    required_cols = [
        "season", "game_id", "team_id", "player_id",
        "games_played_before", "team_games_before", "games_played_ratio_before",
        "pts_prev_avg", "reb_prev_avg", "stl_prev_avg", "blk_prev_avg", "tov_prev_avg",
        "fg_pct_before", "tp_pct_before", "ft_pct_before", "ts_pct_before",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in player_game_features_df: {missing}")

    # Flag de jugadores "core" en el momento de este partido
    df["is_core_before"] = (df["games_played_ratio_before"] >= core_threshold).astype(int)

    # Peso para promedios ponderados: qué tan titular es el jugador
    w = df["games_played_ratio_before"].replace(0, np.nan)

    def wavg(x):
        # promedio ponderado ignorando NaN
        return np.nansum(x * w[x.index]) / np.nansum(w[x.index])

    group = df.groupby(["season", "game_id", "team_id"])

    team_game_features = group.agg(
        n_players_game=("player_id", "nunique"),
        n_core_players_before=("is_core_before", "sum"),
        avg_games_played_before=("games_played_before", "mean"),
        avg_games_played_ratio_before=("games_played_ratio_before", "mean"),

        # promedios ponderados por importancia histórica
        pts_prev_avg_w=("pts_prev_avg", wavg),
        reb_prev_avg_w=("reb_prev_avg", wavg),
        stl_prev_avg_w=("stl_prev_avg", wavg),
        blk_prev_avg_w=("blk_prev_avg", wavg),
        tov_prev_avg_w=("tov_prev_avg", wavg),

        fg_pct_before_w=("fg_pct_before", wavg),
        tp_pct_before_w=("tp_pct_before", wavg),
        ft_pct_before_w=("ft_pct_before", wavg),
        ts_pct_before_w=("ts_pct_before", wavg),
    ).reset_index()

    # Rellenar posibles NaN
    num_cols = [
        "avg_games_played_before", "avg_games_played_ratio_before",
        "pts_prev_avg_w", "reb_prev_avg_w", "stl_prev_avg_w", "blk_prev_avg_w", "tov_prev_avg_w",
        "fg_pct_before_w", "tp_pct_before_w", "ft_pct_before_w", "ts_pct_before_w",
    ]
    team_game_features[num_cols] = team_game_features[num_cols].fillna(0.0)

    return team_game_features


# --------------------------------------------------------
# 4) in_game_team_states: estados in-game + merge con pre-game
# --------------------------------------------------------

def _parse_clock_to_seconds(clock_str) -> float:
    """
    Convierte clock tipo 'PT12M00.00S' -> segundos restantes en el periodo.
    """
    if pd.isna(clock_str):
        return np.nan
    s = str(clock_str).strip()
    if not s:
        return np.nan

    # Formato típico: 'PT12M00.00S'
    # quitamos 'PT' y 'S'
    s = s.replace("PT", "").replace("S", "")
    if "M" in s:
        minutes_str, seconds_str = s.split("M")
        try:
            minutes = float(minutes_str)
        except ValueError:
            minutes = 0.0
        try:
            seconds = float(seconds_str)
        except ValueError:
            seconds = 0.0
        return minutes * 60 + seconds
    else:
        # fallback raro
        try:
            return float(s)
        except ValueError:
            return np.nan


def build_in_game_team_states(
    pbp: pd.DataFrame,
    team_game_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un dataset de estados dentro del partido a nivel equipo,
    tomando como estados las filas donde cambia el marcador (más el inicio de cada partido).

    Para cada estado, incluye:
    - Identificadores: season, game_id, period, seconds_remaining_period
    - Marcador: h_score, a_score, score_diff
    - home_team_id, away_team_id
    - home_win (target)
    - Features in-game acumuladas hasta ese estado (por equipo)
    - Features pre-partido (team_game_features_df) para home y away

    pbp se espera con columnas tipo (como tu pbp2023.csv):
    - 'gameid', 'season', 'period', 'clock', 'h_pts', 'a_pts', 'team', 'type', 'desc', ...

    team_game_features_df se espera con columnas:
    - 'game_id', 'team_id', ... (lo que ya generaste con build_team_game_features)
    """

    df = pbp.copy()

    # Aseguramos tipos básicos
    df["gameid"] = df["gameid"].astype(str)
    df["team"] = df["team"].fillna("").astype(str)

    # Rellenar marcador hacia adelante dentro de cada partido
    df = df.sort_values(["gameid"]).reset_index(drop=True)
    df["h_pts"] = df["h_pts"].ffill().fillna(0).astype(float)
    df["a_pts"] = df["a_pts"].ffill().fillna(0).astype(float)

    # ============================
    # 1. Stats de equipo por evento
    # ============================
    for col in ["type", "subtype", "desc"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Tiros de campo
    shot_mask = df["type"].isin(["Made Shot", "Missed Shot"])
    is_made = df["type"] == "Made Shot"
    is_3pt = shot_mask & df["desc"].str.contains("3PT", na=False)

    df["team_fga_evt"] = shot_mask.astype(int)
    df["team_fgm_evt"] = is_made.astype(int)
    df["team_tpa_evt"] = is_3pt.astype(int)
    df["team_tpm_evt"] = (is_made & is_3pt).astype(int)

    # Tiros libres
    ft_mask = df["type"] == "Free Throw"
    df["team_fta_evt"] = ft_mask.astype(int)
    df["team_ftm_evt"] = (ft_mask & ~df["desc"].str.startswith("MISS")).astype(int)

    # Pérdidas
    df["team_tov_evt"] = (df["type"] == "Turnover").astype(int)

    # Puntos del evento (2/3 pts + libres)
    df["team_fg2m_evt"] = df["team_fgm_evt"] - df["team_tpm_evt"]
    df["team_pts_evt"] = (
        2 * df["team_fg2m_evt"] +
        3 * df["team_tpm_evt"] +
        df["team_ftm_evt"]
    )

    # =====================================================
    # 2. Determinar home_team / away_team y home_win por game
    # =====================================================
    # Totales de puntos por equipo en el juego (a partir del PBP)
    team_totals = (
        df[df["team"] != ""]
        .groupby(["gameid", "team"], as_index=False)["team_pts_evt"]
        .sum()
        .rename(columns={"team_pts_evt": "team_pts_final"})
    )

    # Marcador final por game desde columnas h_pts, a_pts
    final_scores = (
        df.groupby("gameid", as_index=False)
        .agg(final_h_pts=("h_pts", "max"), final_a_pts=("a_pts", "max"))
    )

    game_map = team_totals.merge(final_scores, on="gameid", how="left")

    game_map["is_home"] = game_map["team_pts_final"] == game_map["final_h_pts"]
    game_map["is_away"] = game_map["team_pts_final"] == game_map["final_a_pts"]

    home_map = (
        game_map[game_map["is_home"]][["gameid", "team"]]
        .rename(columns={"team": "home_team_id"})
    )
    away_map = (
        game_map[game_map["is_away"]][["gameid", "team"]]
        .rename(columns={"team": "away_team_id"})
    )

    game_teams = home_map.merge(away_map, on="gameid", how="inner")

    # outcome (home_win)
    final_scores["home_win"] = (final_scores["final_h_pts"] > final_scores["final_a_pts"]).astype(int)

    game_info = game_teams.merge(
        final_scores[["gameid", "home_win"]],
        on="gameid",
        how="left",
    )

    # Añadimos home/away + home_win al PBP
    df = df.merge(game_info, on="gameid", how="left")

    # ============================
    # 3. Stats acumuladas home/away dentro del partido
    # ============================
    # Aportación de cada evento a home o away
    df["is_home_event"] = df["team"] == df["home_team_id"]
    df["is_away_event"] = df["team"] == df["away_team_id"]

    def contrib(col_evt, mask):
        return np.where(mask, df[col_evt], 0)

    df["h_pts_evt"] = contrib("team_pts_evt", df["is_home_event"])
    df["a_pts_evt"] = contrib("team_pts_evt", df["is_away_event"])

    df["h_fga_evt"] = contrib("team_fga_evt", df["is_home_event"])
    df["a_fga_evt"] = contrib("team_fga_evt", df["is_away_event"])

    df["h_tpa_evt"] = contrib("team_tpa_evt", df["is_home_event"])
    df["a_tpa_evt"] = contrib("team_tpa_evt", df["is_away_event"])

    df["h_fta_evt"] = contrib("team_fta_evt", df["is_home_event"])
    df["a_fta_evt"] = contrib("team_fta_evt", df["is_away_event"])

    df["h_ftm_evt"] = contrib("team_ftm_evt", df["is_home_event"])
    df["a_ftm_evt"] = contrib("team_ftm_evt", df["is_away_event"])

    df["h_tov_evt"] = contrib("team_tov_evt", df["is_home_event"])
    df["a_tov_evt"] = contrib("team_tov_evt", df["is_away_event"])

    # Cumsum por game (acumulado hasta cada evento)
    g_game = df.groupby("gameid", sort=False)

    for side in ["h", "a"]:
        df[f"{side}_pts_so_far"] = g_game[f"{side}_pts_evt"].cumsum()
        df[f"{side}_fga_so_far"] = g_game[f"{side}_fga_evt"].cumsum()
        df[f"{side}_tpa_so_far"] = g_game[f"{side}_tpa_evt"].cumsum()
        df[f"{side}_fta_so_far"] = g_game[f"{side}_fta_evt"].cumsum()
        df[f"{side}_ftm_so_far"] = g_game[f"{side}_ftm_evt"].cumsum()
        df[f"{side}_tov_so_far"] = g_game[f"{side}_tov_evt"].cumsum()

        # TS% y eFG% in-game
        denom_ts = 2 * (df[f"{side}_fga_so_far"] + 0.44 * df[f"{side}_fta_so_far"])
        df[f"{side}_ts_pct_so_far"] = np.where(
            denom_ts > 0,
            df[f"{side}_pts_so_far"] / denom_ts,
            0.0,
        )

        df[f"{side}_efg_pct_so_far"] = np.where(
            df[f"{side}_fga_so_far"] > 0,
            (df[f"{side}_pts_so_far"] - df[f"{side}_ftm_so_far"]) / (2 * df[f"{side}_fga_so_far"]),
            0.0,
        )

        # % FGA de 3
        df[f"{side}_pct_fga_3pt_so_far"] = np.where(
            df[f"{side}_fga_so_far"] > 0,
            df[f"{side}_tpa_so_far"] / df[f"{side}_fga_so_far"],
            0.0,
        )

    # ============================
    # 4. Definir "estados" dentro del partido
    # ============================
    df["h_score_int"] = df["h_pts"].astype(int)
    df["a_score_int"] = df["a_pts"].astype(int)

    g = df.groupby("gameid", sort=False)
    df["is_first_event"] = g.cumcount() == 0

    df["prev_h_score"] = g["h_score_int"].shift(1)
    df["prev_a_score"] = g["a_score_int"].shift(1)

    # Para el primer evento, igualamos prev al valor actual
    first_mask = df["is_first_event"]
    df.loc[first_mask, "prev_h_score"] = df.loc[first_mask, "h_score_int"]
    df.loc[first_mask, "prev_a_score"] = df.loc[first_mask, "a_score_int"]

    score_change = (
        (df["h_score_int"] != df["prev_h_score"]) |
        (df["a_score_int"] != df["prev_a_score"])
    )

    state_mask = df["is_first_event"] | score_change

    states = df[state_mask].copy()

    # ============================
    # 5. Features de tiempo
    # ============================
    states["seconds_remaining_period"] = states["clock"].apply(_parse_clock_to_seconds)

    # ============================
    # 6. Merge con features pre-partido (team_game_features_df)
    # ============================
    states = states.rename(columns={"gameid": "game_id"})
    states["game_id"] = states["game_id"].astype(str)
    states["home_team_id"] = states["home_team_id"].astype(str)
    states["away_team_id"] = states["away_team_id"].astype(str)

    tgf = team_game_features_df.copy()
    tgf["game_id"] = tgf["game_id"].astype(str)
    tgf["team_id"] = tgf["team_id"].astype(str)

    # home
    h_tgf = tgf.add_prefix("h_")
    states = states.merge(
        h_tgf,
        left_on=["game_id", "home_team_id"],
        right_on=["h_game_id", "h_team_id"],
        how="left",
    )

    # away
    a_tgf = tgf.add_prefix("a_")
    states = states.merge(
        a_tgf,
        left_on=["game_id", "away_team_id"],
        right_on=["a_game_id", "a_team_id"],
        how="left",
    )

    # ============================
    # 7. Columnas finales
    # ============================
    states["score_diff"] = states["h_score_int"] - states["a_score_int"]

    keep_cols = [
        # identificadores
        "season", "game_id", "period", "clock", "seconds_remaining_period",
        "home_team_id", "away_team_id",
        # marcador
        "h_score_int", "a_score_int", "score_diff",
        # target
        "home_win",
        # in-game stats home
        "h_pts_so_far", "h_fga_so_far", "h_tpa_so_far", "h_fta_so_far",
        "h_tov_so_far", "h_ts_pct_so_far", "h_efg_pct_so_far", "h_pct_fga_3pt_so_far",
        # in-game stats away
        "a_pts_so_far", "a_fga_so_far", "a_tpa_so_far", "a_fta_so_far",
        "a_tov_so_far", "a_ts_pct_so_far", "a_efg_pct_so_far", "a_pct_fga_3pt_so_far",
    ]

    # añadimos algunas pre-game que seguro tienes en team_game_features_df
    # (ajusta según tus columnas reales)
    pre_game_cols_home = [
        "h_n_players_game", "h_n_core_players_before",
        "h_avg_games_played_before", "h_avg_games_played_ratio_before",
        "h_pts_prev_avg_w", "h_reb_prev_avg_w", "h_ts_pct_before_w",
    ]
    pre_game_cols_away = [
        "a_n_players_game", "a_n_core_players_before",
        "a_avg_games_played_before", "a_avg_games_played_ratio_before",
        "a_pts_prev_avg_w", "a_reb_prev_avg_w", "a_ts_pct_before_w",
    ]

    for c in pre_game_cols_home + pre_game_cols_away:
        if c in states.columns:
            keep_cols.append(c)

    keep_cols = [c for c in keep_cols if c in states.columns]

    states_final = states[keep_cols].copy()

    return states_final


# --------------------------------------------------------
# 5) build_season_states: orquesta todo para UNA temporada
# --------------------------------------------------------

def build_season_states(season: int, data_dir: str = "../data/raw") -> pd.DataFrame:
    """
    Construye el dataset de estados in-game + features pre-partido
    para una temporada dada (ej: 2002, 2010, 2023, ...).

    Lee el archivo data_dir/f"pbp{season}.csv"
    y devuelve un DataFrame con:
        - estado del partido (score_diff, tiempo, etc.)
        - stats in-game de equipo hasta cada estado
        - stats de fuerza pre-partido de equipo
        - target home_win
    """

    path = f"{data_dir}/pbp{season}.csv"
    pbp = pd.read_csv(path)

    # 1) jugador-partido (desde PBP)
    games_df = build_games_df_from_pbp(pbp)

    # 2) jugador-partido con históricos before
    player_game_features_df = build_player_game_features(games_df)

    # 3) equipo-partido pre-game
    team_game_features_df = build_team_game_features(player_game_features_df)

    # 4) estados dentro del partido (cambios de marcador + inicio),
    #    con stats in-game + merge con pre-game
    states_df = build_in_game_team_states(pbp, team_game_features_df)

    # 5) segundos restantes de partido (aprox 4Q * 12min)
    states_df = states_df.copy()
    states_df["seconds_remaining_game"] = (
        (4 - states_df["period"]) * 720 + states_df["seconds_remaining_period"]
    )
    states_df["seconds_remaining_game"] = states_df["seconds_remaining_game"].clip(lower=0)

    return states_df


__all__ = [
    "build_games_df_from_pbp",
    "build_player_game_features",
    "build_team_game_features",
    "build_in_game_team_states",
    "build_season_states",
]

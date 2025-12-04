import pandas as pd
import numpy as np


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

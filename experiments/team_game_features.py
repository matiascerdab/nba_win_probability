import pandas as pd
import numpy as np


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

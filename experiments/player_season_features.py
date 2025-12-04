import pandas as pd
import numpy as np


def build_player_season_features(
    games_df: pd.DataFrame,
    core_threshold: float = 0.6,
    deep_bench_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Agrega stats jugador-partido (games_df) a nivel jugador-temporada.

    games_df se espera con columnas:
    ['season', 'game_id', 'team_id', 'player_id',
     'pts', 'reb', 'stl', 'blk', 'tov', 'pf',
     'fga', 'fgm', 'tpa', 'tpm', 'fta', 'ftm']

    Devuelve un DataFrame con, entre otras:
    - games_played
    - team_games_played
    - games_played_ratio (proxy de "minutos posibles jugados")
    - pts_per_game, reb_per_game, etc.
    - fg_pct, tp_pct, ft_pct, ts_pct
    - is_core_player, is_deep_bench
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

    # 1) Partidos jugados por equipo (por temporada)
    team_games = (
        df[["season", "team_id", "game_id"]]
        .drop_duplicates()
        .groupby(["season", "team_id"], as_index=False)
        .agg(team_games_played=("game_id", "nunique"))
    )

    # 2) Agregación jugador-temporada-equipo
    agg_dict = {
        "game_id": "nunique",
        "pts": "sum",
        "reb": "sum",
        "stl": "sum",
        "blk": "sum",
        "tov": "sum",
        "pf": "sum",
        "fga": "sum",
        "fgm": "sum",
        "tpa": "sum",
        "tpm": "sum",
        "fta": "sum",
        "ftm": "sum",
    }

    players = (
        df.groupby(["season", "team_id", "player_id"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"game_id": "games_played"})
    )

    # 3) Unir team_games_played
    players = players.merge(
        team_games,
        on=["season", "team_id"],
        how="left",
        validate="many_to_one",
    )

    # 4) Ratio partidos jugados / partidos del equipo
    players["games_played_ratio"] = players["games_played"] / players["team_games_played"]

    # 5) Stats por partido
    g = players["games_played"].replace(0, np.nan)

    players["pts_per_game"] = players["pts"] / g
    players["reb_per_game"] = players["reb"] / g
    players["stl_per_game"] = players["stl"] / g
    players["blk_per_game"] = players["blk"] / g

    # Porcentajes de tiro
    players["fg_pct"] = np.where(players["fga"] > 0, players["fgm"] / players["fga"], 0.0)
    players["tp_pct"] = np.where(players["tpa"] > 0, players["tpm"] / players["tpa"], 0.0)
    players["ft_pct"] = np.where(players["fta"] > 0, players["ftm"] / players["fta"], 0.0)

    # True Shooting %
    denom_ts = 2 * (players["fga"] + 0.44 * players["fta"])
    players["ts_pct"] = np.where(denom_ts > 0, players["pts"] / denom_ts, 0.0)

    # 6) Flags de core player / deep bench basados en games_played_ratio
    players["is_core_player"] = (players["games_played_ratio"] >= core_threshold).astype(int)
    players["is_deep_bench"] = (players["games_played_ratio"] <= deep_bench_threshold).astype(int)

    return players

import pandas as pd

from games_df import build_games_df_from_pbp
from player_game_features import build_player_game_features
from team_game_features import build_team_game_features
from in_game_team_states import build_in_game_team_states


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

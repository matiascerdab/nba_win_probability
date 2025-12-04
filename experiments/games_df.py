import pandas as pd


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

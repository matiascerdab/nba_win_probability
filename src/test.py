import pandas as pd

df = pd.read_parquet("data/processed/nba_pbp_features.parquet")

# A nivel partido
games = (
    df.groupby(["season", "game_id", "home_team", "away_team"], as_index=False)
      .agg(has_odds=("home_ml", lambda x: x.notna().any()))
)

coverage_by_season = (
    games.groupby("season")["has_odds"]
    .mean()
    .mul(100)
    .round(1)
)

print("Cobertura de partidos con odds por temporada (%):")
print(coverage_by_season)

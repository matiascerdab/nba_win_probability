from pathlib import Path
import pandas as pd

from features_engine import build_season_states
from odds_utils import load_odds_clean

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "nba_pbp_features.parquet"


def infer_seasons(raw_dir: Path = RAW_DIR) -> list[int]:
    """Busca archivos pbp*.csv en data/raw y devuelve la lista de temporadas."""
    seasons: list[int] = []
    for f in sorted(raw_dir.glob("pbp*.csv")):
        stem = f.stem  # ej: "pbp1997"
        try:
            season = int(stem.replace("pbp", ""))
            seasons.append(season)
        except ValueError:
            continue
    return seasons


def build_full_dataset(seasons: list[int] | None = None) -> pd.DataFrame:
    """Construye el dataset final de estados + odds."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Temporadas a procesar
    if seasons is None:
        seasons = infer_seasons()
    if not seasons:
        raise FileNotFoundError(f"No se encontraron archivos pbp*.csv en {RAW_DIR}")

    # 2) Estados in-game por temporada (sale de features_engine)
    all_states: list[pd.DataFrame] = []

    for season in seasons:
        print(f"Construyendo estados para la temporada {season}...")
        season_df = build_season_states(season=season, data_dir=str(RAW_DIR))

        if "seconds_remaining_game" not in season_df.columns:
            raise KeyError(
                "El DataFrame de estados no contiene 'seconds_remaining_game'. "
                "Revisa build_season_states en features_engine.py."
            )

        total_reg_seconds = 4 * 12 * 60
        season_df = season_df.copy()
        season_df["seconds_elapsed_game"] = total_reg_seconds - season_df["seconds_remaining_game"]
        season_df["seconds_elapsed_game"] = season_df["seconds_elapsed_game"].clip(lower=0)

        all_states.append(season_df)

    df = pd.concat(all_states, ignore_index=True)

    print(
        f"Dataset de estados: {len(df)} filas, "
        f"{df['game_id'].nunique()} partidos, "
        f"{df['season'].nunique()} temporadas."
    )

    # 3) Cargar fechas de partido desde PBP con fecha: pbpXXXX_with_date.csv
    game_dates_frames: list[pd.DataFrame] = []

    for season in seasons:
        pbp_with_date_path = PROCESSED_DIR / f"pbp{season}_with_date.csv"
        if not pbp_with_date_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo con fecha para la temporada {season}: "
                f"{pbp_with_date_path}. "
                "Asegúrate de haber corrido src/add_game_dates.py antes."
            )

        # Sólo necesitamos gameid + game_date
        pbp = pd.read_csv(pbp_with_date_path, usecols=["gameid", "game_date"])

        tmp = (
            pbp.assign(season=season)
               .rename(columns={"gameid": "game_id"})
        )
        tmp["game_id"] = tmp["game_id"].astype(str)
        tmp = tmp.drop_duplicates(subset=["season", "game_id", "game_date"])

        game_dates_frames.append(tmp)

    game_dates = pd.concat(game_dates_frames, ignore_index=True)
    game_dates["game_date"] = pd.to_datetime(game_dates["game_date"]).dt.normalize()

    print(f"[build_dataset] game_dates: {len(game_dates)} registros únicos de (season, game_id, game_date).")

    # 4) Merge de fechas al nivel de estados (season + game_id)
    df["game_id"] = df["game_id"].astype(str)
    df = df.merge(
        game_dates,
        on=["season", "game_id"],
        how="left",
        validate="m:1",
    )

    if df["game_date"].isna().any():
        missing_games = df.loc[df["game_date"].isna(), "game_id"].nunique()
        print(f"[build_dataset] WARNING: {missing_games} partidos sin 'game_date' tras el merge.")

    # 5) Normalizar columnas de equipo para el merge con odds
    #    Usamos los IDs de equipo que salen de features_engine (home_team_id/away_team_id)
    if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
        raise KeyError(
            "El DataFrame de estados no contiene 'home_team_id'/'away_team_id'. "
            "Revisa build_in_game_team_states en features_engine.py."
        )

    df["home_team"] = df["home_team_id"].astype(str)
    df["away_team"] = df["away_team_id"].astype(str)

    # 6) Cargar odds ya limpias (con home_team/away_team en abreviación PBP)
    print("[build_dataset] Cargando odds y realizando merge...")
    odds = load_odds_clean()

    # 7) Merge estados + odds por (season, game_date, home_team, away_team)
    df = df.merge(
        odds,
        on=["season", "game_date", "home_team", "away_team"],
        how="left",
        validate="m:1",
    )

    missing = df["home_ml"].isna().mean() * 100
    print(f"[build_dataset] {missing:.1f}% de filas sin odds (home_ml nulo).")

    print(f"Guardando dataset final en {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)
    print("Listo ✅")

    return df


def main() -> None:
    build_full_dataset()


if __name__ == "__main__":
    main()

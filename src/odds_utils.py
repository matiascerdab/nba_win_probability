from pathlib import Path
import pandas as pd
import numpy as np


def american_to_prob(ml: pd.Series) -> pd.Series:
    """
    Convierte cuotas tipo moneyline americano a probabilidad implícita.

    ml < 0: favorito
    ml > 0: underdog
    """
    ml = pd.to_numeric(ml, errors="coerce")

    p = np.where(
        (ml < 0) & ~np.isnan(ml),
        (-ml) / ((-ml) + 100.0),
        np.where(
            (ml > 0) & ~np.isnan(ml),
            100.0 / (ml + 100.0),
            np.nan,
        ),
    )
    return pd.Series(p, index=ml.index)


# Mapa de nombres en oddsData.csv -> abreviaciones usadas en el PBP
ODDS_TO_PBP_TEAM: dict[str, str] = {
    "atl": "ATL",
    "bkn": "BRK",
    "bos": "BOS",
    "cha": "CHA",
    "chi": "CHI",
    "cle": "CLE",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gs": "GSW",
    "hou": "HOU",
    "ind": "IND",
    "lac": "LAC",
    "lal": "LAL",
    "mem": "MEM",
    "mia": "MIA",
    "mil": "MIL",
    "min": "MIN",
    "no": "NOP",
    "ny": "NYK",
    "okc": "OKC",
    "orl": "ORL",
    "phi": "PHI",
    "phx": "PHX",
    "por": "POR",
    "sa": "SAS",
    "sac": "SAC",
    "tor": "TOR",
    "utah": "UTA",
    "wsh": "WAS",
}


def _map_team_name(name: str) -> str:
    name = str(name).strip()
    if name not in ODDS_TO_PBP_TEAM:
        raise ValueError(
            f"Nombre de equipo en oddsData.csv no reconocido: {name!r}.\n"
            f"Añádelo a ODDS_TO_PBP_TEAM en odds_utils.py."
        )
    return ODDS_TO_PBP_TEAM[name]


def load_odds_clean(path: str | Path | None = None) -> pd.DataFrame:
    """
    Lee el CSV de cuotas (dataset de Kaggle 'nba-betting-data-october-2007-to-june-2024')
    y devuelve un DataFrame limpio a nivel partido con columnas:

        season, game_date, home_team, away_team,
        home_ml, away_ml, p_home_book, p_away_book,
        spread_full_game, total_full_game,
        spread_2H, total_2H

    Estructura esperada del CSV (ej. nba_2008-2025.csv):
        - 'season'
        - 'date'
        - 'home', 'away'
        - 'whos_favored'   ('home' / 'away')
        - 'spread'         (siempre positivo, spread del favorito)
        - 'total'
        - 'moneyline_home', 'moneyline_away'
        - 'h2_spread', 'h2_total'   (segunda mitad)
    """
    project_root = Path(__file__).resolve().parents[1]

    if path is None:
        odds_dir = project_root / "data" / "external" / "nba_odds"
        csvs = sorted(odds_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                "No se encontraron CSV de odds en "
                f"{odds_dir}. Descarga primero el dataset de Kaggle."
            )
        # Tomamos el primero (o ajusta a nombre concreto si quieres)
        path = csvs[0]

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró el archivo de odds en la ruta indicada:\n"
            f"  {path}\n\n"
            "Verifica que exista ese archivo o pasa la ruta correcta a "
            "load_odds_clean(path='ruta/al/archivo.csv')."
        )

    print(f"[odds_utils] Cargando odds desde: {path}")
    odds_raw = pd.read_csv(path)

    required_cols_raw = [
        "season", "date", "home", "away",
        "whos_favored", "spread", "total",
        "moneyline_home", "moneyline_away",
    ]
    missing_raw = [c for c in required_cols_raw if c not in odds_raw.columns]
    if missing_raw:
        raise ValueError(
            "El archivo de odds no tiene las columnas mínimas esperadas.\n"
            f"Faltan columnas: {missing_raw}\n"
            f"Columnas presentes: {list(odds_raw.columns)}"
        )

    # Normalizar tipos básicos
    odds_raw["game_date"] = pd.to_datetime(odds_raw["date"]).dt.normalize()
    odds_raw["season"] = odds_raw["season"].astype(int)

    # Mapear equipos Kaggle -> códigos PBP
    odds_raw["home_team"] = odds_raw["home"].map(_map_team_name)
    odds_raw["away_team"] = odds_raw["away"].map(_map_team_name)

    if odds_raw["home_team"].isna().any() or odds_raw["away_team"].isna().any():
        bad = odds_raw[odds_raw["home_team"].isna() | odds_raw["away_team"].isna()]
        unknown = sorted(set(bad["home"].tolist()) | set(bad["away"].tolist()))
        raise ValueError(
            "Hay equipos en el CSV de odds que no se pudieron mapear.\n"
            f"Añádelos a ODDS_TO_PBP_TEAM: {unknown}"
        )

    # Moneylines
    odds = odds_raw[
        [
            "season",
            "game_date",
            "home_team",
            "away_team",
            "moneyline_home",
            "moneyline_away",
            "whos_favored",
            "spread",
            "total",
        ]
    ].copy()

    odds = odds.rename(
        columns={
            "moneyline_home": "home_ml",
            "moneyline_away": "away_ml",
        }
    )

    fav = odds["whos_favored"].astype(str).str.strip().str.lower()
    spread = pd.to_numeric(odds["spread"], errors="coerce")

    # Spread full game DESDE EL PUNTO DE VISTA DEL HOME:
    # - si el favorito es el home: home -spread  => spread_full_game = -spread
    # - si el favorito es el away: home +spread  => spread_full_game = +spread
    odds["spread_full_game"] = np.where(
        fav == "home",
        -spread,
        spread,
    )

    odds["total_full_game"] = pd.to_numeric(odds["total"], errors="coerce")

    # 2nd half spread/total (si existen)
    if "h2_spread" in odds_raw.columns:
        h2_spread = pd.to_numeric(odds_raw["h2_spread"], errors="coerce")
        odds["spread_2H"] = np.where(
            fav == "home",
            -h2_spread,
            h2_spread,
        )
    else:
        odds["spread_2H"] = np.nan

    if "h2_total" in odds_raw.columns:
        odds["total_2H"] = pd.to_numeric(odds_raw["h2_total"], errors="coerce")
    else:
        odds["total_2H"] = np.nan

    # Probabilidades implícitas de moneyline (partido completo)
    odds["p_home_raw"] = american_to_prob(odds["home_ml"])
    odds["p_away_raw"] = american_to_prob(odds["away_ml"])
    total_prob = odds["p_home_raw"] + odds["p_away_raw"]
    total_prob = total_prob.replace(0, np.nan)

    odds["p_home_book"] = odds["p_home_raw"] / total_prob
    odds["p_away_book"] = odds["p_away_raw"] / total_prob

    # Un solo registro por partido
    odds = odds.drop_duplicates(
        subset=["season", "game_date", "home_team", "away_team"],
        keep="first",
    )

    cols_out = [
        "season",
        "game_date",
        "home_team",
        "away_team",
        "home_ml",
        "away_ml",
        "p_home_book",
        "p_away_book",
        "spread_full_game",
        "total_full_game",
        "spread_2H",
        "total_2H",
    ]

    return odds[cols_out]

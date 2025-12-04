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
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Jersey": "NJN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Seattle": "SEA",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS",
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
    Lee el CSV de cuotas (oddsData.csv) y devuelve un DataFrame limpio
    a nivel partido con columnas:

        season, game_date, home_team, away_team,
        home_ml, away_ml, p_home_book, p_away_book

    Estructura esperada del CSV original (oddsData.csv):
        - 'date'
        - 'season'
        - 'team'
        - 'home/visitor'   ('vs' = local, '@' = visita)
        - 'opponent'
        - 'moneyLine'
        - 'opponentMoneyLine'
    """
    # Raíz del proyecto (carpeta que contiene 'src')
    project_root = Path(__file__).resolve().parents[1]

    if path is None:
        # Ruta por defecto que me indicaste
        path = project_root / "data" / "external" / "nba_odds" / "oddsData.csv"

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
        "date",
        "season",
        "team",
        "home/visitor",
        "opponent",
        "moneyLine",
        "opponentMoneyLine",
    ]
    missing_raw = [c for c in required_cols_raw if c not in odds_raw.columns]
    if missing_raw:
        raise ValueError(
            "El archivo de odds no tiene las columnas esperadas.\n"
            f"Faltan columnas: {missing_raw}\n"
            f"Columnas presentes en el archivo: {list(odds_raw.columns)}"
        )

    # Normalizar tipos básicos
    odds_raw["game_date"] = pd.to_datetime(odds_raw["date"]).dt.normalize()
    odds_raw["season"] = odds_raw["season"].astype(int)

    # Tomamos sólo las filas donde el 'team' es local: 'vs'
    hv = odds_raw["home/visitor"].astype(str).str.strip().str.lower()
    is_home = hv == "vs"

    odds_home = odds_raw[is_home].copy()

    # Mapear nombres de equipos (en oddsData) -> abreviaciones PBP
    odds_home["home_team"] = odds_home["team"].map(_map_team_name)
    odds_home["away_team"] = odds_home["opponent"].map(_map_team_name)

    # Verificación rápida de que no haya nombres sin mapear
    if odds_home["home_team"].isna().any() or odds_home["away_team"].isna().any():
        bad_rows = odds_home[odds_home["home_team"].isna() | odds_home["away_team"].isna()]
        unknown = sorted(
            set(bad_rows["team"].tolist()) | set(bad_rows["opponent"].tolist())
        )
        raise ValueError(
            "Hay equipos en oddsData.csv que no se pudieron mapear a códigos PBP.\n"
            f"Revísalos y añádelos a ODDS_TO_PBP_TEAM: {unknown}"
        )

    # Construimos DataFrame partido-nivel
    odds = odds_home[
        ["season", "game_date", "home_team", "away_team", "moneyLine", "opponentMoneyLine"]
    ].copy()
    odds = odds.rename(
        columns={
            "moneyLine": "home_ml",
            "opponentMoneyLine": "away_ml",
        }
    )

    # Probabilidades implícitas
    odds["p_home_raw"] = american_to_prob(odds["home_ml"])
    odds["p_away_raw"] = american_to_prob(odds["away_ml"])
    total = odds["p_home_raw"] + odds["p_away_raw"]
    total = total.replace(0, np.nan)

    odds["p_home_book"] = odds["p_home_raw"] / total
    odds["p_away_book"] = odds["p_away_raw"] / total

    # Por seguridad, un solo registro por partido
    odds = odds.drop_duplicates(
        subset=["season", "game_date", "home_team", "away_team"],
        keep="first",
    )

    return odds[
        [
            "season",
            "game_date",
            "home_team",
            "away_team",
            "home_ml",
            "away_ml",
            "p_home_book",
            "p_away_book",
        ]
    ]

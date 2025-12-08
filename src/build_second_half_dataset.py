# src/build_second_half_dataset.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features_engine import build_second_half_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "nba_odds"


def load_states() -> pd.DataFrame:
    """
    Carga el dataset de estados in-game que generas con build_dataset.py
    (nba_pbp_features.parquet).
    """
    path = PROCESSED_DIR / "nba_pbp_features.parquet"
    print(f"[build_second_half_dataset] Cargando estados desde: {path}")
    df = pd.read_parquet(path)
    return df


# ===========================
# 2H SPREAD: CARGAR DEL CSV
# ===========================

def load_second_half_spreads() -> pd.DataFrame:
    """
    Carga el CSV 'nba_2008-2025.csv' que contiene:
      - spread (partido completo)
      - h2_spread (second half spread, en magnitud)
      - whos_favored ("home"/"away")
      - away, home (códigos en minúscula), date, season, etc.

    A partir de eso construimos:
      - home_team / away_team en el MISMO formato que el PBP
      - pregame_spread_home  (spread del partido desde POV del home, con signo)
      - h2_home_spread       (second half spread desde POV del home, con signo)
    """
    path = EXTERNAL_DIR / "nba_2008-2025.csv"
    print(f"[build_second_half_dataset] Cargando 2H spreads desde: {path}")
    odds = pd.read_csv(path)

    # Renombramos columnas básicas
    odds = odds.rename(
        columns={
            "date": "game_date",
            "home": "home_team",
            "away": "away_team",
        }
    )

    # Tipos básicos
    odds["season"] = odds["season"].astype(int)
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.normalize()

    # === MAPE0 DE CODIGOS DE ODDS -> CODIGOS PBP ===

    def map_odds_team_to_pbp(code: str, season: int) -> str:
        """
        Mapea códigos de odds (atl, sa, gs, no, bkn, wsh, ny, etc.)
        a los códigos usados en el PBP (ATL, SAS, GSW, NOH/NOP, NJN/BRK, WAS, NYK, ...).
        """
        code = str(code).strip().lower()

        # casos especiales con cambios de franquicia/nombre
        if code == "bkn":
            # Nets: New Jersey (NJN) -> Brooklyn (BRK)
            # Ajusta estos años según lo que veas en tu base_2h CSV:
            return "NJN" if season < 2013 else "BRK"

        if code == "no":
            # New Orleans Hornets/Pelicans
            # 2008–2013: seguramente NOH
            # 2014+: NOP
            return "NOH" if season < 2014 else "NOP"

        if code == "wsh":
            return "WAS"

        if code == "gs":
            return "GSW"

        if code == "ny":
            return "NYK"

        # mapeo directo "normal"
        base_map = {
            "atl": "ATL",
            "bos": "BOS",
            "cha": "CHA",
            "chi": "CHI",
            "cle": "CLE",
            "dal": "DAL",
            "den": "DEN",
            "det": "DET",
            "hou": "HOU",
            "ind": "IND",
            "lac": "LAC",
            "lal": "LAL",
            "mem": "MEM",
            "mia": "MIA",
            "mil": "MIL",
            "min": "MIN",
            "okc": "OKC",
            "orl": "ORL",
            "phi": "PHI",
            "phx": "PHX",
            "por": "POR",
            "sa": "SAS",
            "sac": "SAC",
            "tor": "TOR",
            "utah": "UTA",
        }

        if code in base_map:
            return base_map[code]

        raise ValueError(f"[map_odds_team_to_pbp] Equipo desconocido en odds: '{code}' (season={season})")

    # aplicar el mapeo a home_team y away_team
    odds["home_team"] = [
        map_odds_team_to_pbp(c, s) for c, s in zip(odds["home_team"], odds["season"])
    ]
    odds["away_team"] = [
        map_odds_team_to_pbp(c, s) for c, s in zip(odds["away_team"], odds["season"])
    ]

    # Spread PRE-PARTIDO desde el punto de vista del home:
    odds["pregame_spread_home"] = np.where(
        odds["whos_favored"].str.lower() == "home",
        -odds["spread"],
        odds["spread"],
    )

    # Second half spread desde POV del home:
    odds["h2_home_spread"] = np.where(
        odds["whos_favored"].str.lower() == "home",
        -odds["h2_spread"],
        odds["h2_spread"],
    )

    keep_cols = [
        "season",
        "game_date",
        "home_team",
        "away_team",
        "pregame_spread_home",
        "h2_home_spread",
    ]
    odds = odds[keep_cols].drop_duplicates()

    return odds


def inspect_datasets(base_2h: pd.DataFrame, spreads_2h: pd.DataFrame) -> None:
    """
    Imprime info básica de las dos tablas antes del merge
    para revisar que las claves coinciden.
    También calcula cuántas claves coinciden exactamente.
    """
    print("\n=== [INSPECCIÓN DE BASES ANTES DEL MERGE] ===")

    print("\nbase_2h.shape:", base_2h.shape)
    print("spreads_2h.shape:", spreads_2h.shape)

    print("\nColumnas base_2h:")
    print(base_2h.columns.tolist())

    print("\nColumnas spreads_2h:")
    print(spreads_2h.columns.tolist())

    # Ejemplos de claves
    print("\nEjemplo base_2h (season, game_date, home_team, away_team):")
    print(base_2h[["season", "game_date", "home_team", "away_team"]].head(10))

    print("\nEjemplo spreads_2h (season, game_date, home_team, away_team):")
    print(spreads_2h[["season", "game_date", "home_team", "away_team"]].head(10))

    # Revisar tipos
    print("\nTipos base_2h:")
    print(base_2h[["season", "game_date", "home_team", "away_team"]].dtypes)

    print("\nTipos spreads_2h:")
    print(spreads_2h[["season", "game_date", "home_team", "away_team"]].dtypes)

    # Chequear si las claves coinciden
    merge_keys = ["season", "game_date", "home_team", "away_team"]
    left_keys = base_2h[merge_keys].drop_duplicates()
    right_keys = spreads_2h[merge_keys].drop_duplicates()
    overlap = pd.merge(left_keys, right_keys, on=merge_keys, how="inner")
    print(f"\nCoincidencias exactas en claves de merge: {len(overlap)}")
    print(f"Filas únicas base_2h: {len(left_keys)}, spreads_2h: {len(right_keys)}")

    print("=== FIN INSPECCIÓN ===\n")


def build_second_half_dataset(save_debug_csv: bool = True) -> pd.DataFrame:
    # 1) Cargar estados (nba_pbp_features.parquet)
    states = load_states()

    # 2) Construir frame de 2º tiempo (half-time + margen final)
    base_2h = build_second_half_frame(states)  # viene de features_engine.py

    # Normalizar nombres de equipo
    if "home_team_id" in base_2h.columns and "home_team" not in base_2h.columns:
        base_2h = base_2h.rename(
            columns={
                "home_team_id": "home_team",
                "away_team_id": "away_team",
            }
        )

    # 3) Cargar spreads de 2ª mitad desde nba_2008-2025.csv
    spreads_2h = load_second_half_spreads()

    # === Guardar CSVs completos para inspección ===
    if save_debug_csv:
        base_path = PROCESSED_DIR / "second_half_base_2h_full.csv"
        spreads_path = PROCESSED_DIR / "second_half_spreads_2h_full.csv"
        base_2h.to_csv(base_path, index=False)
        spreads_2h.to_csv(spreads_path, index=False)
        print(f"[build_second_half_dataset] Guardado base_2h en: {base_path}")
        print(f"[build_second_half_dataset] Guardado spreads_2h en: {spreads_path}")

    # === Imprimir resumen de qué tan bien matchean las claves ===
    inspect_datasets(base_2h, spreads_2h)

    # === Claves de merge ===
    merge_keys = ["season", "game_date", "home_team", "away_team"]

    # Asegurar unicidad por partido en ambos lados
    base_2h = base_2h.drop_duplicates(subset=merge_keys, keep="first")
    spreads_2h = spreads_2h.drop_duplicates(subset=merge_keys, keep="first")

    # 4) Merge 1:1 partido ↔ línea de mercado
    df = base_2h.merge(
        spreads_2h,
        on=merge_keys,
        how="inner",
        validate="1:1",
    )

    # 5) Residuo del mercado en 2H
    df["resid_2H"] = df["margin_2H_real"] - df["h2_home_spread"]

    print("[build_second_half_dataset] Ejemplo de columnas del dataset final:")
    print(df.head(3).T)

    return df


def main():
    df = build_second_half_dataset(save_debug_csv=True)

    out_path = PROCESSED_DIR / "nba_second_half_dataset.parquet"
    print(f"[build_second_half_dataset] Guardando dataset 2H en: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("[build_second_half_dataset] Listo ✅")


if __name__ == "__main__":
    main()

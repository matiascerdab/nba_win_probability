from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ruta al root del proyecto
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_second_half_dataset.parquet"

# Archivo de debug donde vamos a volcar una muestra para ver a mano
DEBUG_CSV_PATH = ROOT / "data" / "processed" / "nba_second_half_debug_season_gt_2014.csv"

# Si quieres inspeccionar un partido concreto, pon aquí el game_id (como string) o déjalo en None
GAME_ID_TO_INSPECT = None   # ejemplo: "41900001"


def main():
    print("[test] Cargando dataset de segundo tiempo desde:")
    print("   ", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {DATA_PATH}.\n"
            "Primero corre: python src/build_second_half_dataset.py"
        )

    df = pd.read_parquet(DATA_PATH)

    # --- Filtrar solo seasons > 2014 ---
    if "season" not in df.columns:
        raise KeyError("El dataset no tiene columna 'season', no puedo filtrar por temporada.")

    df = df[df["season"] > 2014].copy()

    print(f"[test] Filtrado a seasons > 2014. Filas restantes: {len(df)}")
    print(f"[test] Temporadas presentes: {sorted(df['season'].unique())}")

    required_cols = ["margin_2H_real", "h2_home_spread"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Faltan columnas necesarias en el parquet:\n"
            f"  {missing}"
        )

    # ======================================================
    # 1) CREAR CSV DE DEBUG CON COLUMNAS CLAVE
    # ======================================================
    cols_debug = [
        "season",
        "game_id",
        "game_date",      # si existe
        "home_team",
        "away_team",
        "margin_HT",
        "margin_final",
        "margin_2H_real",
        "pregame_spread_home",
        "h2_home_spread",
    ]

    cols_debug = [c for c in cols_debug if c in df.columns]  # por si falta game_date

    df_debug = df[cols_debug].sort_values(["season", "game_id"]).copy()

    # si quieres solo una muestra, puedes cortar:
    # df_debug = df_debug.head(500)

    DEBUG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_debug.to_csv(DEBUG_CSV_PATH, index=False)
    print("\n[test] CSV de debug guardado en:")
    print("   ", DEBUG_CSV_PATH)
    print("[test] Ábrelo en Excel / LibreOffice para ver partidos uno por uno.")

    # ======================================================
    # 2) OPCIONAL: IMPRIMIR UN PARTIDO CONCRETO (por game_id)
    # ======================================================
    if GAME_ID_TO_INSPECT is not None:
        sel = df_debug[df_debug["game_id"].astype(str) == str(GAME_ID_TO_INSPECT)]
        print(f"\n[test] Partidos con game_id = {GAME_ID_TO_INSPECT}:")
        if sel.empty:
            print("  (ninguno encontrado en seasons > 2014)")
        else:
            print(sel.to_string(index=False))

    # ======================================================
    # 3) DISTRIBUCIÓN COVER / NO COVER / PUSH (como antes)
    # ======================================================
    margin = df["margin_2H_real"].to_numpy(dtype=float)
    spread = df["h2_home_spread"].to_numpy(dtype=float)

    edge = margin + spread  # margen handicapado del lado home

    cover_label = np.where(
        edge > 0,
        "home_cubre",
        np.where(edge < 0, "home_no_cubre", "push"),
    )

    df["cover_label"] = cover_label
    df["edge_2H"] = edge

    counts = df["cover_label"].value_counts().reindex(
        ["home_cubre", "home_no_cubre", "push"]
    )
    props = df["cover_label"].value_counts(normalize=True).reindex(
        ["home_cubre", "home_no_cubre", "push"]
    )

    print("\n=== DISTRIBUCIÓN COVER 2H (GLOBAL, SEASONS > 2014) ===")
    print("Conteos:")
    print(counts)
    print("\nProporciones:")
    print(props.round(3))

    # Gráfico de barras
    fig, ax = plt.subplots()
    labels = ["home_cubre", "home_no_cubre", "push"]
    vals = [props.get(l, 0.0) for l in labels]

    ax.bar(labels, vals)
    ax.set_ylabel("Proporción")
    ax.set_title("Distribución 2H spread (seasons > 2014)\nhome cubre / no cubre / push")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")

    plt.tight_layout()

    # Histograma edge
    fig2, ax2 = plt.subplots()
    ax2.hist(edge, bins=31)
    ax2.axvline(0, linestyle="--")
    ax2.set_xlabel("edge_2H = margin_2H_real + h2_home_spread")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Histograma edge 2H (seasons > 2014)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

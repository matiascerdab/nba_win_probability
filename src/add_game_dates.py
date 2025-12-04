import pandas as pd
from pathlib import Path

# === Paths ===
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# === Cargamos Games.csv ===
games = pd.read_csv(raw_dir / "Games.csv", low_memory=False)
games["gameId"] = games["gameId"].astype(str)

# Igualamos formato con el de los PBP: "00" + zfill(8)
games["gameId_formatted"] = games["gameId"].apply(lambda x: "00" + x.zfill(8))

# Extraemos solo la fecha (sin hora)
games["game_date"] = pd.to_datetime(
    games["gameDateTimeEst"], errors="coerce"
).dt.date

games = games[["gameId_formatted", "game_date"]].drop_duplicates()

print(f"Archivo Games.csv cargado con {len(games)} partidos con fecha.")

# === Iteramos sobre los pbp ===
pbp_files = sorted(raw_dir.glob("pbp*.csv"))
if not pbp_files:
    raise FileNotFoundError("⚠️ No se encontraron archivos pbp*.csv en data/raw/")

for pbp_file in pbp_files:
    print(f"\nProcesando {pbp_file.name}...")

    pbp = pd.read_csv(pbp_file, low_memory=False)
    pbp["gameId_formatted"] = pbp["gameid"].astype(str).apply(lambda x: "00" + x.zfill(8))

    merged = pbp.merge(
        games,
        on="gameId_formatted",
        how="left"
    )

    out_path = processed_dir / pbp_file.name.replace(".csv", "_with_date.csv")
    merged.to_csv(out_path, index=False)

    missing = merged["game_date"].isna().sum()
    print(f"✅ Guardado: {out_path} ({len(merged)} filas, {missing} sin fecha)")

print("\n🎯 Listo. Todos los PBP ahora tienen 'game_date' con solo la fecha (sin hora).")

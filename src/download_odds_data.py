# This script is used to download NBA odds data from Kaggle

import os
import subprocess


def download_nba_odds():
    """
    Download the Kaggle dataset with NBA odds
    and unzip it into data/external/nba_odds
    """

    # Dataset de Kaggle con moneylines, spreads, etc.
    dataset_slug = "christophertreasure/nba-odds-data"

    # Carpeta donde se guardarán los CSV de cuotas
    output_dir = "data/external/nba_odds"

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", output_dir,
        "--unzip",
    ]

    print("Downloading NBA odds dataset from Kaggle:")
    print("  ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Download completed. Files in:", output_dir)


if __name__ == "__main__":
    download_nba_odds()

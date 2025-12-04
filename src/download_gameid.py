#This script it's used to download the data from Kaggle

import os
import subprocess

def download_nba_gameid():
    "download the Kaggle dataset NBA play by play data 1997-2023"
    "to the folder data/raw and then unfiled from zip"

    dataset_slug = "eoinamoore/historical-nba-data-and-player-box-scores"
    output_dir = "data/raw"

    os.makedirs(output_dir, exist_ok=True)

    cmd=[
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", output_dir,
        "--unzip"
     ]
    
    print("Downloading dataset from Kaggle", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Download completed. Files in:",output_dir)

if __name__ == "__main__":
    download_nba_gameid()
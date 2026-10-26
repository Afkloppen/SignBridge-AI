
import pandas as pd
from pathlib import Path

for split in ["train", "val", "test"]:
    path = Path("data/raw") / f"{split}_metadata.csv"

    df = pd.read_csv(path)

    print(f"\n{'=' * 40}")
    print(f"{split.upper()} DATA")
    print(f"{'=' * 40}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 entries:")
    print(df[["label", "video_path", "include_50"]].head())
    print("\nINCLUDE-50 entries:", df["include_50"].sum())

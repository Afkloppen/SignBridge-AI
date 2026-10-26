
import pandas as pd
import requests
from pathlib import Path

root = Path("data/raw")

# Get the actual archive list from Zenodo.
response = requests.get(
    "https://zenodo.org/api/records/4010759",
    timeout=60,
)
response.raise_for_status()

files = response.json().get("files", [])
archive_names = [
    item["key"] for item in files
    if item["key"].lower().endswith(".zip")
]

# Find which video categories are needed by INCLUDE-50.
frames = [
    pd.read_csv(root / f"{split}_metadata.csv")
    for split in ["train", "val", "test"]
]
data = pd.concat(frames, ignore_index=True)
data = data[data["include_50"] == True]

categories = sorted({
    str(path).split("/")[0]
    for path in data["video_path"]
})

print("INCLUDE-50 video categories:")
for category in categories:
    matches = [
        name for name in archive_names
        if name.lower().startswith(category.lower() + "_")
    ]
    print(f"\n{category}:")
    for name in matches:
        item = next(f for f in files if f["key"] == name)
        print(f"  {name} — {item['size'] / (1024**3):.2f} GB")

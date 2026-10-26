
import pandas as pd
import requests
from pathlib import Path
from remotezip import RemoteZip
import time

ROOT = Path("data/raw")
OUTPUT = Path("data/videos")
RECORD_URL = "https://zenodo.org/api/records/4010759"
MANIFEST_PATH = Path("data/sample_manifest.csv")

# Load the three metadata splits
train = pd.read_csv(ROOT / "train_metadata.csv")
val = pd.read_csv(ROOT / "val_metadata.csv")
test = pd.read_csv(ROOT / "test_metadata.csv")

# Select signs that have enough examples in every split
counts = pd.concat(
    [
        train["label"].value_counts().rename("train"),
        val["label"].value_counts().rename("val"),
        test["label"].value_counts().rename("test"),
    ],
    axis=1,
).fillna(0).astype(int)

eligible = counts[
    (counts["train"] >= 5)
    & (counts["val"] >= 2)
    & (counts["test"] >= 2)
].sort_values("train", ascending=False)

labels = eligible.head(10).index.tolist()

if not labels:
    raise SystemExit("No suitable sign classes found in all splits.")

print("Selected signs:")
for label in labels:
    print(f"  {label}: {counts.loc[label].to_dict()}")

# Select examples for each split
selected = []
for split_name, frame, limit in [
    ("train", train, 16),
    ("val", val, 2),
    ("test", test, 2),
]:
    for label in labels:
        rows = frame[frame["label"] == label].head(limit)
        for _, row in rows.iterrows():
            selected.append({
                "split": split_name,
                "label": label,
                "video_path": row["video_path"],
            })

# Read the official archive listing
session = requests.Session()
response = session.get(RECORD_URL, timeout=45)
response.raise_for_status()
archives = [
    item for item in response.json()["files"]
    if item["key"].lower().endswith(".zip")
]

# Reuse existing manifest information when possible
old_manifest = {}
if MANIFEST_PATH.exists():
    old = pd.read_csv(MANIFEST_PATH)
    for _, row in old.iterrows():
        old_manifest[row["video_path"]] = row.to_dict()

# Cache ZIP listings while processing the same archive part
zip_cache = {}
downloaded = set()

def find_video(relative_path):
    category = relative_path.split("/")[0].lower()
    candidates = [
        item for item in archives
        if item["key"].lower().startswith(category + "_")
    ]

    for archive in candidates:
        key = archive["key"]

        try:
            if key not in zip_cache:
                print("Inspecting archive:", key)
                remote = RemoteZip(archive["links"]["self"], timeout=45)
                names = remote.namelist()
                zip_cache[key] = (remote, names)

            remote, names = zip_cache[key]
            matches = [
                name for name in names
                if name.replace("\\", "/").endswith(relative_path)
            ]

            if not matches:
                continue

            destination = OUTPUT / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            print("Downloading:", relative_path)
            with remote.open(matches[0]) as source:
                with destination.open("wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)

            return destination.exists() and destination.stat().st_size > 0

        except Exception as error:
            print(f"Archive error ({key}): {error}")
            time.sleep(1)

    return False

# Download missing videos only
for index, item in enumerate(selected, start=1):
    relative_path = item["video_path"]
    destination = OUTPUT / relative_path

    if destination.exists() and destination.stat().st_size > 0:
        downloaded.add(relative_path)
        print(f"[{index}/{len(selected)}] Already available: {relative_path}")
        continue

    try:
        if find_video(relative_path):
            downloaded.add(relative_path)
        else:
            print("NOT DOWNLOADED:", relative_path)
    except Exception as error:
        print("Failed:", relative_path, error)

# Close remote ZIP connections
for remote, _ in zip_cache.values():
    try:
        remote.close()
    except Exception:
        pass

# Save the updated manifest
manifest = []
for item in selected:
    destination = OUTPUT / item["video_path"]
    manifest.append({
        **item,
        "local_path": str(destination),
        "downloaded": destination.exists() and destination.stat().st_size > 0,
    })

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)

print("\n========== DOWNLOAD SUMMARY ==========")
print("Selected videos:", len(selected))
print("Available videos:", sum(item["downloaded"] for item in manifest))
print("Manifest:", MANIFEST_PATH)

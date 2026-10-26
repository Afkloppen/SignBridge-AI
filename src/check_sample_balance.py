
import pandas as pd
from pathlib import Path

manifest = pd.read_csv("data/sample_manifest.csv")

# Count only videos that actually exist on disk.
manifest["exists"] = manifest["local_path"].apply(
    lambda path: Path(path).is_file()
)

available = manifest[manifest["exists"]].copy()

print("TOTAL AVAILABLE VIDEOS:", len(available))
print("\nVIDEO COUNTS PER SIGN AND SPLIT:\n")

counts = (
    available.groupby(["label", "split"])
    .size()
    .unstack(fill_value=0)
)

for split in ["train", "val", "test"]:
    if split not in counts.columns:
        counts[split] = 0

counts = counts[["train", "val", "test"]]
print(counts.to_string())

# A preliminary minimum for a usable prototype experiment.
usable = counts[
    (counts["train"] >= 5)
    & (counts["val"] >= 1)
    & (counts["test"] >= 1)
]

print("\nCLASSES WITH AT LEAST 5 TRAIN, 1 VAL, AND 1 TEST VIDEO:")
print(usable.to_string() if not usable.empty else "None yet")

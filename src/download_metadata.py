
from pathlib import Path
from datasets import load_dataset

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading INCLUDE dataset metadata...")

dataset = load_dataset("ai4bharat/INCLUDE")

for split_name, split_data in dataset.items():
    df = split_data.to_pandas()

    # Keep only INCLUDE-50 entries for our first experiment.
    if "include_50" in df.columns:
        df = df[df["include_50"] == True]

    output_file = output_dir / f"{split_name}_metadata.csv"
    df.to_csv(output_file, index=False)

    print(f"{split_name}: {len(df)} rows saved to {output_file}")

print("Metadata download complete!")

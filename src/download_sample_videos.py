
import pandas as pd
import requests
from pathlib import Path
from remotezip import RemoteZip

df = pd.read_csv("data/raw/train_metadata.csv")
video_path = df.loc[0, "video_path"]
category = video_path.split("/")[0]

print("Looking for:", video_path)

response = requests.get(
    "https://zenodo.org/api/records/4010759",
    timeout=30
)
response.raise_for_status()
files = response.json()["files"]

archives = [
    f for f in files
    if f["key"].lower().endswith(".zip")
    and f["key"].lower().startswith(category.lower() + "_")
]

for archive in archives:
    print("\nChecking:", archive["key"])
    try:
        with RemoteZip(archive["links"]["self"], timeout=30) as remote:
            matches = [
                name for name in remote.namelist()
                if name.replace("\\", "/").endswith(video_path)
            ]

            if not matches:
                print("Not in this part.")
                continue

            target = matches[0]
            output = Path("data/videos") / video_path
            output.parent.mkdir(parents=True, exist_ok=True)

            print("Found video! Downloading only this file...")
            with remote.open(target) as source, output.open("wb") as dest:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    dest.write(chunk)

            print("SUCCESS:", output)
            break

    except Exception as error:
        print("Could not check this part:", error)
else:
    print("\nVideo not located. No more downloads attempted.")

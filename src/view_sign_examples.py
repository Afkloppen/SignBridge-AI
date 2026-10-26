
from pathlib import Path

import cv2
import pandas as pd

manifest_path = Path("data/sample_manifest.csv")

if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest not found: {manifest_path}")

df = pd.read_csv(manifest_path)
available = df[df["downloaded"].astype(str).str.lower() == "true"]

print("\nSigns with downloaded example videos:")
for label in sorted(available["label"].dropna().unique()):
    count = (available["label"] == label).sum()
    print(f"  {label} ({count} videos)")

label = input("\nType a sign exactly as listed above: ").strip()
matches = available[available["label"] == label]

if matches.empty:
    print("No downloaded video found for that label.")
    raise SystemExit(0)

video_path = Path(matches.iloc[0]["local_path"])

if not video_path.exists():
    print("Video file not found:", video_path)
    raise SystemExit(1)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Could not open video:", video_path)
    raise SystemExit(1)

print("Playing:", label)
print("Press Q in the video window to close it.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.putText(
        frame, label, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.imshow("SignBridge AI - Training Sign Example", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

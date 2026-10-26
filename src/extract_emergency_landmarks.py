
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

ROOT = Path("external_datasets/emergency_isl/extracted")
OUTPUT = Path("data/processed/emergency_landmarks.npz")
NUM_FRAMES = 16

LABELS = [
    "accident", "call", "doctor", "help",
    "hot", "lose", "pain", "thief"
]


def hand_features(hand):
    if hand is None:
        return np.zeros(63, dtype=np.float32)

    return np.array([
        value
        for landmark in hand.landmark
        for value in (landmark.x, landmark.y, landmark.z)
    ], dtype=np.float32)


def extract_video(path, holistic):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total < 1:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []
    current = 0
    wanted = set(indices.tolist())

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if current in wanted:
            result = holistic.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            features = np.concatenate([
                hand_features(result.left_hand_landmarks),
                hand_features(result.right_hand_landmarks),
            ])
            frames.append((current, features))

        current += 1

    cap.release()

    # Preserve temporal order and fill missing frames if needed.
    found = dict(frames)
    output = []

    for index in indices:
        if index in found:
            output.append(found[index])
        elif output:
            output.append(output[-1])
        else:
            output.append(np.zeros(126, dtype=np.float32))

    return np.asarray(output, dtype=np.float32)


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

    X, y, paths = [], [], []
    counts = {}

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for label in LABELS:
            folder = ROOT / f"{label}_Raw"
            videos = sorted(folder.rglob("*.avi")) if folder.exists() else []
            counts[label] = 0

            print(f"\n{label}: {len(videos)} videos")

            for video in videos:
                sequence = extract_video(video, holistic)

                if sequence is None or sequence.shape != (NUM_FRAMES, 126):
                    print(f"Skipped unreadable video: {video.name}")
                    continue

                X.append(sequence)
                y.append(label)
                paths.append(str(video))
                counts[label] += 1

                if len(X) % 25 == 0:
                    print(f"Processed {len(X)} videos")

    if not X:
        raise RuntimeError(
            "No videos processed. Check the extracted folder structure."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y),
        video_paths=np.asarray(paths),
        labels=np.asarray(LABELS),
    )

    print("\nDataset extraction complete!")
    print("Saved to:", OUTPUT.resolve())
    print("Feature shape:", np.asarray(X).shape)
    print("Total videos processed:", len(X))
    print("Counts:", counts)


if __name__ == "__main__":
    main()

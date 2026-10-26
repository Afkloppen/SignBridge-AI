
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import Counter

# Project paths
MANIFEST = Path("data/sample_manifest.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "landmarks.npz"

# Keep a fixed number of frames per video
NUM_FRAMES = 16
NUM_FEATURES = 21 * 3 * 2  # 2 hands, 21 landmarks, x/y/z

def get_hand_features(hand_landmarks):
    """Return 21 hand landmarks as 63 numerical features."""
    if hand_landmarks is None:
        return np.zeros(21 * 3, dtype=np.float32)

    return np.array(
        [
            coordinate
            for landmark in hand_landmarks.landmark
            for coordinate in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )

def extract_video(video_path, holistic):
    """Sample 16 frames and extract left/right hand landmarks."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0, frame_count - 1, NUM_FRAMES, dtype=int
    )

    sequence = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            sequence.append(
                np.zeros(NUM_FEATURES, dtype=np.float32)
            )
            continue

        # MediaPipe requires RGB frames
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb_frame)

        left_hand = get_hand_features(result.left_hand_landmarks)
        right_hand = get_hand_features(result.right_hand_landmarks)

        sequence.append(
            np.concatenate([left_hand, right_hand])
        )

    cap.release()
    return np.stack(sequence).astype(np.float32)

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    manifest = pd.read_csv(MANIFEST)

    required = {"label", "split", "local_path", "downloaded"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    # Process only videos that are marked downloaded and exist locally
    usable = manifest[
        manifest["downloaded"].astype(str).str.lower().isin(
            ["true", "1"]
        )
    ].copy()

    usable = usable[
        usable["local_path"].apply(
            lambda path: Path(str(path)).is_file()
        )
    ]

    print("Videos available for processing:", len(usable))

    if usable.empty:
        print("No local videos found. Check data/sample_manifest.csv.")
        return

    features = []
    labels = []
    splits = []
    video_paths = []

    # Create one MediaPipe model and reuse it for all videos
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for number, (_, row) in enumerate(
            usable.iterrows(), start=1
        ):
            path = Path(str(row["local_path"]))

            try:
                sequence = extract_video(path, holistic)

                if sequence is None:
                    print(f"[{number}] Skipped unreadable video: {path}")
                    continue

                features.append(sequence)
                labels.append(str(row["label"]))
                splits.append(str(row["split"]))
                video_paths.append(str(path))

                print(
                    f"[{number}/{len(usable)}] Processed: "
                    f"{row['label']} ({row['split']})"
                )

            except Exception as error:
                print(f"Error processing {path}: {error}")

    if not features:
        print("No features were extracted.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=str)
    split_array = np.array(splits, dtype=str)
    paths_array = np.array(video_paths, dtype=str)

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        splits=split_array,
        video_paths=paths_array,
    )

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Videos processed:", len(X))
    print("Feature array shape:", X.shape)
    print("Features per frame:", NUM_FEATURES)
    print("Sign classes:", len(set(y)))
    print("Videos per split:", dict(Counter(split_array)))
    print("Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()

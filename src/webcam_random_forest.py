
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import joblib

MODEL_PATH = Path("models/signbridge_random_forest.joblib")
NUM_FRAMES = 16
CONFIDENCE_THRESHOLD = 0.65


def hand_features(hand):
    if hand is None:
        return np.zeros(63, dtype=np.float32)

    return np.array(
        [
            coordinate
            for landmark in hand.landmark
            for coordinate in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Random Forest model missing. Run "
            "python src\\train_random_forest.py first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]

    if saved["num_frames"] != NUM_FRAMES:
        raise ValueError("Model and webcam frame counts do not match.")

    speech = pyttsx3.init()
    speech.setProperty("rate", 155)

    sequence = deque(maxlen=NUM_FRAMES)
    recent_predictions = deque(maxlen=5)

    current_word = "Show a sign"
    confidence = 0.0
    last_spoken = None
    last_speech_time = 0.0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        speech.stop()
        raise RuntimeError("Could not open webcam. Try camera index 1.")

    print("Random Forest webcam started.")
    print("Press Q to quit or C to reset speech.")

    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:

            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read webcam frame.")
                    break

                # Extract landmarks from the original frame.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                left = hand_features(result.left_hand_landmarks)
                right = hand_features(result.right_hand_landmarks)
                sequence.append(np.concatenate([left, right]))

                if len(sequence) == NUM_FRAMES:
                    features = np.stack(sequence).astype(np.float32)
                    flattened = features.reshape(1, -1)
                    scaled = scaler.transform(flattened)

                    probabilities = model.predict_proba(scaled)[0]
                    predicted_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_index])
                    predicted_label = str(model.classes_[predicted_index])

                    # Always show the raw prediction for debugging.
                    current_word = predicted_label
                    recent_predictions.append(predicted_label)

                    stable = (
                        len(recent_predictions) == recent_predictions.maxlen
                        and len(set(recent_predictions)) == 1
                    )

                    # Speak only stable, confident predictions.
                    if stable and confidence >= CONFIDENCE_THRESHOLD:
                        now = time.time()

                        if (
                            current_word != last_spoken
                            and now - last_speech_time >= 1.5
                        ):
                            speech.say(current_word)
                            speech.runAndWait()
                            last_spoken = current_word
                            last_speech_time = time.time()

                # Display a mirrored image; model input stays unflipped.
                display = cv2.flip(frame, 1)

                cv2.putText(
                    display,
                    f"Prediction: {current_word}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Confidence: {confidence:.0%}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Frames collected: {len(sequence)}/{NUM_FRAMES}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "Q: quit | C: reset speech",
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("SignBridge AI - Random Forest Debug", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    last_spoken = None
                    last_speech_time = 0.0
                    recent_predictions.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()
        speech.stop()


if __name__ == "__main__":
    main()

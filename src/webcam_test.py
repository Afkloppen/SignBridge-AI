
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from torch import nn

MODEL_PATH = Path("models/signbridge_lstm.pt")
LABELS_PATH = Path("models/labels.json")
NUM_FRAMES = 16

class SignLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.classifier(self.dropout(sequence[:, -1, :]))

def hand_features(hand):
    if hand is None:
        return np.zeros(63, dtype=np.float32)
    return np.array(
        [
            value
            for landmark in hand.landmark
            for value in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )

def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Model files missing. Run src/train_model.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    id_to_label = {int(index): label for label, index in labels.items()}

    model = SignLSTM(
        checkpoint["input_size"],
        checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

    speech = pyttsx3.init()
    speech.setProperty("rate", 155)

    sequence = deque(maxlen=NUM_FRAMES)
    recent_predictions = deque(maxlen=5)
    last_spoken = None
    current_word = "Show a sign to the camera"
    confidence = 0.0
    last_speech_time = 0.0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or try camera index 1."
        )

    print("Webcam started. Press Q to quit; C to clear the spoken-word lock.")

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
                    print("Could not read from webcam.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                frame = cv2.flip(frame, 1)

                left = hand_features(result.left_hand_landmarks)
                right = hand_features(result.right_hand_landmarks)
                sequence.append(np.concatenate([left, right]))

                if len(sequence) == NUM_FRAMES:
                    features = np.stack(sequence).astype(np.float32)
                    features = (features - mean) / std
                    tensor = torch.tensor(
                        features[None, ...], dtype=torch.float32, device=device
                    )

                    with torch.no_grad():
                        probabilities = torch.softmax(model(tensor), dim=1)[0]
                        confidence, prediction = probabilities.max(dim=0)
                        confidence = float(confidence.item())
                        predicted_id = int(prediction.item())

                    # Only show a prediction after repeated agreement.
                    recent_predictions.append(predicted_id)
                    stable = (
                        len(recent_predictions) == recent_predictions.maxlen
                        and len(set(recent_predictions)) == 1
                    )

                    if stable and confidence >= 0.70:
                        current_word = id_to_label[predicted_id]

                        # Avoid speaking the same prediction repeatedly.
                        now = time.time()
                        if (
                            current_word != last_spoken
                            and now - last_speech_time >= 1.5
                        ):
                            speech.say(current_word)
                            speech.runAndWait()
                            last_spoken = current_word
                            last_speech_time = time.time()
                    elif confidence < 0.70:
                        current_word = "Uncertain sign"

                cv2.putText(
                    frame,
                    f"Prediction: {current_word}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.0%}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Q: quit | C: allow speaking the same word again",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("SignBridge AI - Webcam Test", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("c"):
                    last_spoken = None
                    recent_predictions.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()
        speech.stop()

if __name__ == "__main__":
    main()


import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from torch import nn

MODEL_PATH = Path("models/signbridge_cnn_lstm.pt")
LABELS_PATH = Path("models/cnn_lstm_labels.json")
NUM_FRAMES = 16
CONFIDENCE_THRESHOLD = 0.50


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


class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(self.dropout(sequence[:, -1, :]))


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "CNN+LSTM model files missing. Run "
            "python src\\train_cnn_lstm.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    labels = __import__("json").loads(
        LABELS_PATH.read_text(encoding="utf-8")
    )
    id_to_label = {int(index): label for label, index in labels.items()}

    model = CNNLSTM(
        checkpoint["input_size"],
        checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

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

    print("CNN + LSTM webcam started. Press Q to quit, C to reset speech.")

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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                left = hand_features(result.left_hand_landmarks)
                right = hand_features(result.right_hand_landmarks)
                sequence.append(np.concatenate([left, right]))

                if len(sequence) == NUM_FRAMES:
                    features = np.stack(sequence).astype(np.float32)
                    features = (features - mean) / std
                    tensor = torch.tensor(
                        features[None, ...],
                        dtype=torch.float32,
                        device=device,
                    )

                    with torch.no_grad():
                        probabilities = torch.softmax(model(tensor), dim=1)[0]
                        confidence_tensor, prediction = probabilities.max(dim=0)
                        confidence = float(confidence_tensor.item())
                        predicted_id = int(prediction.item())

                    predicted_label = id_to_label[predicted_id]
                    current_word = predicted_label
                    recent_predictions.append(predicted_label)

                    stable = (
                        len(recent_predictions) == recent_predictions.maxlen
                        and len(set(recent_predictions)) == 1
                    )

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

                display = cv2.flip(frame, 1)
                cv2.putText(
                    display, f"Prediction: {current_word}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    display, f"Confidence: {confidence:.0%}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
                )
                cv2.putText(
                    display, f"Frames: {len(sequence)}/{NUM_FRAMES}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                cv2.imshow("SignBridge AI - CNN + LSTM", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    recent_predictions.clear()
                    last_spoken = None
                    last_speech_time = 0.0

    finally:
        camera.release()
        cv2.destroyAllWindows()
        speech.stop()


if __name__ == "__main__":
    main()

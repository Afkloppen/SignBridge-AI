
import json
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "signbridge_cnn_lstm.pt"
LABELS_PATH = BASE_DIR / "models" / "cnn_lstm_labels.json"

NUM_FRAMES = 16
NUM_FEATURES = 126

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SignBridge AI API",
    description=(
        "Hospital communication support using sign recognition "
        "and locally running generative AI."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. CNN-LSTM model
# Keep this architecture consistent with training.
# --------------------------------------------------

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

        return self.classifier(
            self.dropout(sequence[:, -1, :])
        )


# --------------------------------------------------
# 4. Load the trained sign recognition model
# --------------------------------------------------

sign_model = None
sign_labels = {}
sign_mean = None
sign_std = None
model_load_error = None


def load_sign_model():
    global sign_model
    global sign_labels
    global sign_mean
    global sign_std
    global model_load_error

    if not MODEL_PATH.exists():
        model_load_error = f"Model file not found: {MODEL_PATH}"
        print(model_load_error)
        return

    if not LABELS_PATH.exists():
        model_load_error = f"Labels file not found: {LABELS_PATH}"
        print(model_load_error)
        return

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            label_data = json.load(file)

        # Expected label format: {"word": class_index}
        if isinstance(label_data, dict):
            sign_labels = {
                int(index): str(label)
                for label, index in label_data.items()
            }
        elif isinstance(label_data, list):
            sign_labels = {
                index: str(label)
                for index, label in enumerate(label_data)
            }
        else:
            raise ValueError("Unsupported labels JSON format.")

        sign_model = CNNLSTM(
            checkpoint["input_size"],
            checkpoint["num_classes"],
        ).to(DEVICE)

        sign_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        sign_model.eval()

        sign_mean = np.asarray(
            checkpoint["mean"],
            dtype=np.float32,
        ).reshape(-1)

        sign_std = np.asarray(
            checkpoint["std"],
            dtype=np.float32,
        ).reshape(-1)

        if (
            len(sign_mean) != NUM_FEATURES
            or len(sign_std) != NUM_FEATURES
        ):
            raise ValueError(
                "Checkpoint normalization must contain 126 features."
            )

        sign_std[sign_std < 1e-6] = 1.0

        model_load_error = None
        print("SignBridge CNN-LSTM model loaded successfully.")

    except Exception as exc:
        sign_model = None
        model_load_error = str(exc)
        print("Model loading error:", repr(exc))


load_sign_model()


# --------------------------------------------------
# 5. Request schemas
# --------------------------------------------------

class SignPredictionRequest(BaseModel):
    frames: list[list[float]]


class ImproveMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "English"


class MessagePreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


# --------------------------------------------------
# 6. Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "app": "SignBridge AI",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "api": "ready",
        "sign_model_exists": MODEL_PATH.exists(),
        "labels_exist": LABELS_PATH.exists(),
        "sign_model_loaded": sign_model is not None,
        "local_ai_model": OLLAMA_MODEL,
        "local_ai_endpoint": OLLAMA_URL,
        "model_load_error": model_load_error,
    }


# --------------------------------------------------
# 7. Sign prediction endpoint
# Input: 16 frames, each containing 126 features.
# --------------------------------------------------

@app.post("/sign/predict")
def predict_sign(request: SignPredictionRequest):
    if sign_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The sign model could not be loaded. "
                f"Details: {model_load_error}"
            ),
        )

    frames = np.asarray(request.frames, dtype=np.float32)

    if frames.shape != (NUM_FRAMES, NUM_FEATURES):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {NUM_FRAMES} frames with "
                f"{NUM_FEATURES} features each. "
                f"Received shape {frames.shape}."
            ),
        )

    if not np.isfinite(frames).all():
        raise HTTPException(
            status_code=422,
            detail="Frames contain invalid numeric values.",
        )

    # Use the same training normalization statistics.
    normalized = (frames - sign_mean) / sign_std

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    try:
        with torch.no_grad():
            logits = sign_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            confidence, predicted_index = torch.max(
                probabilities,
                dim=0,
            )

        class_index = int(predicted_index.item())
        confidence_value = float(confidence.item())

        predicted_label = sign_labels.get(
            class_index,
            "Unknown",
        )

        return {
            "prediction": predicted_label,
            "label": predicted_label,
            "confidence": confidence_value,
            "needs_confirmation": True,
        }

    except Exception as exc:
        print("Sign prediction error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Sign prediction failed.",
        )


# --------------------------------------------------
# 8. Local GenAI using Ollama
# No OpenAI API key or paid API required.
# --------------------------------------------------

@app.post("/genai/improve")
def improve_message(request: ImproveMessageRequest):
    original_text = request.text.strip()

    if not original_text:
        raise HTTPException(
            status_code=422,
            detail="Please enter a message.",
        )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You help Deaf patients communicate with "
                            "healthcare professionals. Rewrite the patient's "
                            "message clearly and respectfully in the requested "
                            "language. Preserve the original meaning. Do not "
                            "invent symptoms, diagnoses, medicines, or facts. "
                            "Do not give medical advice. If the original "
                            "message is ambiguous, do not guess its meaning. "
                            "Return only the improved message."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Requested language: {request.language}\n"
                            f"Patient message: {original_text}"
                        ),
                    },
                ],
                "options": {
                    "temperature": 0.2,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        suggestion = (
            result.get("message", {})
            .get("content", "")
            .strip()
        )

        if not suggestion:
            raise HTTPException(
                status_code=502,
                detail="Local AI returned an empty response.",
            )

        return {
            "original_text": original_text,
            "suggestion": suggestion,
            "needs_confirmation": True,
        }

    except requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. Start Ollama and ensure "
                "the llama3.2:3b model is installed."
            ),
        )

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail=(
                "Local AI took too long to respond. "
                "Please try again."
            ),
        )

    except requests.RequestException as exc:
        print("Ollama request error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail="The local AI request failed.",
        )


# --------------------------------------------------
# 9. Message preview endpoint
# --------------------------------------------------

@app.post("/message/preview")
def preview_message(request: MessagePreviewRequest):
    message = request.text.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Please enter a message.",
        )

    return {
        "original_text": message,
        "message": "Message received. Review it before sending.",
        "needs_confirmation": True,
    }

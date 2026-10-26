
import json
import threading
from collections import deque
from io import BytesIO
from pathlib import Path
from html import escape

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import speech_recognition as sr
import torch
from gtts import gTTS
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)
from torch import nn


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "signbridge_cnn_lstm.pt"
LABELS_PATH = BASE_DIR / "models" / "cnn_lstm_labels.json"

NUM_FRAMES = 16
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="SignBridge AI | Hospital Communication",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# UI STYLING
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(ellipse at 85% 0%, #173653 0%, transparent 42%), #081321;
    color: #f1f5f9;
    font-family: Inter, sans-serif;
}
[data-testid="stHeader"] { background: rgba(8,19,33,.92); }
.block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 3rem; }
h1,h2,h3,p,label { color: #f1f5f9; }

.sb-brand { display:flex; align-items:center; gap:14px; margin-bottom:22px; }
.sb-logo {
    background:linear-gradient(135deg,#0891b2,#2563eb);
    width:55px;height:55px;border-radius:16px;
    display:flex;align-items:center;justify-content:center;font-size:29px;
}
.sb-brand-title { color:#fff;font-size:27px;font-weight:800; }
.sb-brand-subtitle { color:#b4c7db;font-size:13px; }
.sb-hero {
    padding:26px 28px;border:1px solid #31506d;border-radius:20px;
    background:linear-gradient(120deg,#142a42,#102238 65%,#103746);
    margin-bottom:22px;
}
.sb-eyebrow { color:#5eead4;font-size:11px;font-weight:800;letter-spacing:2px;margin-bottom:10px; }
.sb-hero-title { color:#fff;font-size:clamp(27px,3vw,38px);line-height:1.15;font-weight:800; }
.sb-hero-desc { color:#c4d4e5;font-size:14px;line-height:1.7;margin-top:10px; }
.sb-section { color:#fff;font-size:19px;font-weight:750;margin-bottom:6px; }
.sb-muted { color:#afc2d7;font-size:13px;line-height:1.65; }
.sb-card { background:#102137;border:1px solid #2c435d;border-radius:15px;padding:17px;margin:8px 0; }
.sb-prediction { font-size:22px;font-weight:800;color:#67e8f9;overflow-wrap:anywhere; }
.sb-patient {
    background:#123955;border:1px solid #28658b;border-radius:13px;
    padding:13px 15px;margin:9px 0;color:#fff;overflow-wrap:anywhere;
}
.sb-doctor {
    background:#123b35;border:1px solid #267567;border-radius:13px;
    padding:13px 15px;margin:9px 0;color:#fff;overflow-wrap:anywhere;
}
.stButton > button {
    background:#193a58!important;color:#fff!important;border:1px solid #416b8d!important;
    border-radius:10px!important;min-height:43px;font-weight:650!important;
}
.stButton > button p { color:#fff!important; }
.stButton > button:hover { background:#087e8b!important;border-color:#67e8f9!important; }
.stTextArea textarea,.stTextInput input,.stSelectbox [data-baseweb="select"] > div {
    background:#101f34!important;color:#fff!important;border:1px solid #3b526e!important;
    border-radius:10px!important;
}
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#2b4058!important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# CNN + LSTM MODEL
# =========================================================

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
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(self.dropout(sequence[:, -1, :]))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    label_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    id_to_label = {int(index): label for label, index in label_data.items()}

    model = CNNLSTM(
        checkpoint["input_size"], checkpoint["num_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

    return model, device, id_to_label, mean, std


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


# =========================================================
# EMBEDDED WEBCAM PROCESSOR
# =========================================================

class SignVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.bundle = load_model()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.sequence = deque(maxlen=NUM_FRAMES)
        self.recent = deque(maxlen=5)
        self.lock = threading.Lock()
        self.current_word = "Waiting for a sign"
        self.confidence = 0.0
        self.frames_ready = 0
        self.error = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        features = np.concatenate([
            hand_features(result.left_hand_landmarks),
            hand_features(result.right_hand_landmarks),
        ])
        self.sequence.append(features)
        self.frames_ready = len(self.sequence)

        if self.bundle and len(self.sequence) == NUM_FRAMES:
            model, device, labels, mean, std = self.bundle
            x = (np.stack(self.sequence).astype(np.float32) - mean) / std
            tensor = torch.tensor(x[None, ...], dtype=torch.float32, device=device)

            try:
                with torch.no_grad():
                    probabilities = torch.softmax(model(tensor), dim=1)[0]
                    confidence_tensor, prediction = probabilities.max(dim=0)

                word = labels[int(prediction.item())]
                confidence = float(confidence_tensor.item())
                self.recent.append(word)

                stable = (
                    len(self.recent) == self.recent.maxlen
                    and len(set(self.recent)) == 1
                )

                with self.lock:
                    self.current_word = (
                        word if stable and confidence >= CONFIDENCE_THRESHOLD
                        else "Hold sign steady"
                    )
                    self.confidence = confidence
            except Exception as exc:
                self.error = str(exc)

        display = cv2.flip(image, 1)
        with self.lock:
            word = self.current_word
            confidence = self.confidence

        cv2.rectangle(
            display, (10, 10),
            (min(display.shape[1] - 10, 560), 100),
            (10, 25, 43), -1,
        )
        cv2.putText(
            display, f"Sign: {word[:35]}", (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 240, 230), 2,
        )
        cv2.putText(
            display,
            f"Confidence: {confidence:.0%} | Frames: {self.frames_ready}/{NUM_FRAMES}",
            (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (240, 248, 255), 1,
        )
        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def get_prediction(self):
        with self.lock:
            return self.current_word, self.confidence, self.frames_ready


# =========================================================
# TEXT TO SPEECH
# =========================================================

def speak(text, language="en"):
    text = text.strip()
    if not text:
        st.warning("Enter a message first.")
        return

    try:
        audio = BytesIO()
        gTTS(text=text, lang=language).write_to_fp(audio)
        audio.seek(0)
        st.audio(audio, format="audio/mp3")
    except Exception as exc:
        st.error(f"Audio generation failed. Check your internet connection. {exc}")


# =========================================================
# SESSION STATE AND CALLBACKS
# =========================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "editable_draft" not in st.session_state:
    st.session_state.editable_draft = ""

if "reply_input" not in st.session_state:
    st.session_state.reply_input = ""


def add_to_patient_message(text):
    text = text.strip()
    if text:
        current = st.session_state.get("editable_draft", "").strip()
        st.session_state["editable_draft"] = f"{current} {text}".strip()


def add_prediction_callback(word, confidence):
    if (
        word not in ("Waiting for a sign", "Hold sign steady")
        and confidence >= CONFIDENCE_THRESHOLD
    ):
        add_to_patient_message(word)


def send_patient_message_callback():
    message = st.session_state.get("editable_draft", "").strip()
    if message:
        st.session_state.conversation.append({
            "speaker": "Patient",
            "text": message,
        })
        st.session_state["editable_draft"] = ""


def clear_patient_message_callback():
    st.session_state["editable_draft"] = ""


def add_doctor_reply_callback():
    message = st.session_state.get("reply_input", "").strip()
    if message:
        st.session_state.conversation.append({
            "speaker": "Doctor",
            "text": message,
        })
        st.session_state["reply_input"] = ""


def clear_conversation_callback():
    st.session_state.conversation = []


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="sb-brand">
    <div class="sb-logo">🏥</div>
    <div>
        <div class="sb-brand-title">SignBridge AI</div>
        <div class="sb-brand-subtitle">Accessible patient–doctor communication</div>
    </div>
</div>
<div class="sb-hero">
    <div class="sb-eyebrow">HOSPITAL ACCESSIBILITY · SIGN · UNDERSTAND · RESPOND</div>
    <div class="sb-hero-title">One conversation. Patient and doctor connected.</div>
    <div class="sb-hero-desc">
        Recognize supported signs, review patient messages, and let doctors
        reply by voice or text in one shared workspace.
    </div>
</div>
""", unsafe_allow_html=True)

if load_model() is None:
    st.error(
        "Model files missing. Check models/signbridge_cnn_lstm.pt "
        "and models/cnn_lstm_labels.json."
    )
    st.stop()


# =========================================================
# SHARED HOSPITAL WORKSPACE
# =========================================================

left, right = st.columns([1, 1.15], gap="large")


# ---------------- PATIENT SIDE ----------------

with left:
    st.markdown(
        '<div class="sb-section">📷 Patient sign input</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-muted">Start the camera and hold a supported sign steady. '
        'Review each prediction before adding it to a message.</div>',
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="signbridge-hospital-camera",
        video_processor_factory=SignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        async_processing=True,
    )

    st.caption("Allow camera access in your browser.")

    @st.fragment(run_every="1s")
    def live_prediction():
        processor = ctx.video_processor
        if processor is None:
            st.info("Start the camera to begin sign recognition.")
            return

        word, confidence, count = processor.get_prediction()

        st.markdown(
            '<div class="sb-card"><b>Current prediction</b>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sb-prediction">{escape(word)}</div>',
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(confidence, 0.0), 1.0),
            text=f"Model confidence: {confidence:.0%}",
        )
        st.caption(f"Frames collected: {count}/{NUM_FRAMES}")

        st.button(
            "＋ Add prediction to patient message",
            key="add_prediction",
            use_container_width=True,
            disabled=(
                word in ("Waiting for a sign", "Hold sign steady")
                or confidence < CONFIDENCE_THRESHOLD
            ),
            on_click=add_prediction_callback,
            args=(word, confidence),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    live_prediction()

    st.divider()
    st.markdown(
        '<div class="sb-section">🩺 Common hospital messages</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-muted">Optional text shortcuts. These are not '
        'recognized signs; review them before sending.</div>',
        unsafe_allow_html=True,
    )

    quick_messages = [
        "I need help, please.",
        "I am in pain.",
        "I need water.",
        "I feel dizzy.",
        "Please call my family.",
        "I need the washroom.",
    ]

    for index in range(0, len(quick_messages), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            item_index = index + offset
            if item_index < len(quick_messages):
                with col:
                    st.button(
                        quick_messages[item_index],
                        key=f"quick_{item_index}",
                        use_container_width=True,
                        on_click=add_to_patient_message,
                        args=(quick_messages[item_index],),
                    )


# ---------------- DOCTOR / SHARED CONVERSATION ----------------

with right:
    st.markdown(
        '<div class="sb-section">💬 Patient–doctor conversation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-muted">Patient messages and doctor replies stay '
        'together in one conversation history.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**PATIENT MESSAGE · SIGN → TEXT → SPEECH**")
    st.text_area(
        "Patient message",
        key="editable_draft",
        height=100,
        placeholder="Add a sign prediction or type a message...",
        label_visibility="collapsed",
    )

    speak_col, send_col, clear_col = st.columns(3)

    with speak_col:
        if st.button("🔊 Speak", key="speak_patient", use_container_width=True):
            speak(st.session_state.get("editable_draft", ""), "en")

    with send_col:
        st.button(
            "Send message",
            key="send_patient",
            use_container_width=True,
            on_click=send_patient_message_callback,
        )

    with clear_col:
        st.button(
            "Clear",
            key="clear_patient",
            use_container_width=True,
            on_click=clear_patient_message_callback,
        )

    st.divider()

    st.markdown("**DOCTOR'S REPLY · SPEECH → TEXT**")
    st.caption(
        "Record the doctor's speech, transcribe it, review the text, "
        "and add it to the conversation."
    )

    speech_language = st.selectbox(
        "Doctor's speaking language",
        ["English", "Hindi", "Kannada"],
        key="doctor_speech_language",
    )

    language_codes = {
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Kannada": "kn-IN",
    }

    recorded_audio = st.audio_input(
        "Record doctor's spoken reply",
        key="doctor_audio_input",
    )

    if st.button(
        "🎙️ Transcribe doctor's speech",
        key="transcribe_doctor_speech",
        use_container_width=True,
    ):
        if recorded_audio is None:
            st.warning("Record the doctor's reply first.")
        else:
            try:
                recognizer = sr.Recognizer()

                with sr.AudioFile(
                    BytesIO(recorded_audio.getvalue())
                ) as source:
                    audio_data = recognizer.record(source)

                with st.spinner("Converting speech to text..."):
                    transcript = recognizer.recognize_google(
                        audio_data,
                        language=language_codes[speech_language],
                    )

                # The reply text area is created below this code,
                # so assigning the state here is safe.
                st.session_state["reply_input"] = transcript
                st.success("Speech transcribed. Review the text below.")

            except sr.UnknownValueError:
                st.error(
                    "Speech was unclear. Please record the reply again."
                )
            except sr.RequestError:
                st.error(
                    "Online speech recognition is unavailable. "
                    "Check your internet connection."
                )
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

    st.text_area(
        "Doctor's reply — review or edit",
        key="reply_input",
        height=100,
        placeholder="The doctor's spoken words will appear here...",
    )

    doctor_col, read_col = st.columns(2)

    with doctor_col:
        st.button(
            "Add doctor reply",
            key="add_doctor_reply",
            use_container_width=True,
            on_click=add_doctor_reply_callback,
        )

    with read_col:
        reply_language = st.selectbox(
            "Read reply aloud in",
            ["English", "Hindi", "Kannada"],
            key="reply_language",
        )

    if st.button(
        "🔊 Read doctor's reply aloud",
        key="read_doctor_reply",
        use_container_width=True,
    ):
        speak(
            st.session_state.get("reply_input", ""),
            {"English": "en", "Hindi": "hi", "Kannada": "kn"}[reply_language],
        )

    st.divider()
    st.markdown("**CONVERSATION HISTORY**")

    if not st.session_state.conversation:
        st.info(
            "No messages yet. Send a patient message or add a doctor's reply."
        )
    else:
        for item in st.session_state.conversation[-20:]:
            is_patient = item["speaker"] == "Patient"
            css_class = "sb-patient" if is_patient else "sb-doctor"
            icon = "🤟" if is_patient else "🩺"

            st.markdown(
                f'<div class="{css_class}"><b>{icon} '
                f'{escape(item["speaker"])}</b><br>'
                f'{escape(item["text"])}</div>',
                unsafe_allow_html=True,
            )

        st.button(
            "Clear conversation history",
            key="clear_history",
            on_click=clear_conversation_callback,
        )


# =========================================================
# SAFETY NOTICE
# =========================================================

st.divider()
st.markdown("""
<div class="sb-muted">
<b>Prototype notice:</b> The current model recognizes a limited vocabulary
of isolated signs and does not yet translate continuous ISL sentences.
Speech transcription requires an internet connection and can make errors.
Confirm medical information with the patient and healthcare professional.
This prototype is not a replacement for qualified interpreters, clinical
assessment, or emergency services.
</div>
""", unsafe_allow_html=True)

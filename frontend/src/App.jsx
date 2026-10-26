
import { useEffect, useRef, useState } from "react";
import "@mediapipe/holistic";
import "@mediapipe/camera_utils";
import "./App.css";

const API = "http://127.0.0.1:8000";

const quickMessages = [
  "I need help",
  "I am in pain",
  "Please call the doctor",
  "I need water",
  "I feel uncomfortable",
  "This is an emergency",
];

const languageSettings = {
  English: { speech: "en-IN", voice: "en-IN" },
  Hindi: { speech: "hi-IN", voice: "hi-IN" },
  Kannada: { speech: "kn-IN", voice: "kn-IN" },
};

export default function App() {
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const holisticRef = useRef(null);
  const framesRef = useRef([]);
  const predictionBusyRef = useRef(false);

  const [patientMessage, setPatientMessage] = useState("");
  const [doctorReply, setDoctorReply] = useState("");
  const [language, setLanguage] = useState("English");
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [cameraStatus, setCameraStatus] = useState("Camera stopped");
  const [prediction, setPrediction] = useState(null);
  const [cameraError, setCameraError] = useState("");
  const [busy, setBusy] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiNote, setAiNote] = useState("");
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    checkAPI();
    return () => {
      stopCamera();
      recognitionRef.current?.stop();
      window.speechSynthesis?.cancel();
    };
  }, []);

  async function checkAPI() {
    try {
      const response = await fetch(`${API}/health`);
      if (!response.ok) throw new Error("API unavailable");
      setApiStatus("Connected");
    } catch {
      setApiStatus("Offline");
    }
  }

  async function startCamera() {
    if (!videoRef.current || cameraRef.current) return;

    setCameraError("");
    setPrediction(null);
    framesRef.current = [];

    try {
      const holistic = new window.Holistic({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      holistic.onResults(async (results) => {
        if (predictionBusyRef.current) return;

        const handFeatures = (landmarks) => {
          if (!landmarks || landmarks.length !== 21) {
            return Array(63).fill(0);
          }
          return landmarks.flatMap((point) => [
            point.x,
            point.y,
            point.z,
          ]);
        };

        const frame = [
          ...handFeatures(results.leftHandLandmarks),
          ...handFeatures(results.rightHandLandmarks),
        ];

        framesRef.current.push(frame);

        if (framesRef.current.length >= 16) {
          const sequence = framesRef.current.slice(0, 16);
          framesRef.current = [];
          predictionBusyRef.current = true;

          try {
            const response = await fetch(`${API}/sign/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ frames: sequence }),
            });

            const result = await response.json();

            if (!response.ok || result.error) {
              throw new Error(result.error || "Prediction failed");
            }

            setPrediction(result);
          } catch (error) {
            setCameraError(error.message || "Sign prediction failed.");
          } finally {
            predictionBusyRef.current = false;
          }
        }
      });

      holisticRef.current = holistic;

      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          if (holisticRef.current && videoRef.current) {
            await holisticRef.current.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });

      cameraRef.current = camera;
      await camera.start();
      setCameraStatus("Camera running");
    } catch (error) {
      stopCamera();
      setCameraError(
        error.message || "Could not start the webcam. Check camera permissions."
      );
      setCameraStatus("Camera stopped");
    }
  }

  function stopCamera() {
    try {
      cameraRef.current?.stop();
    } catch {
      // Camera may already be stopped.
    }

    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    try {
      holisticRef.current?.close();
    } catch {
      // Holistic may already be closed.
    }

    cameraRef.current = null;
    holisticRef.current = null;
    framesRef.current = [];
    predictionBusyRef.current = false;
    setCameraStatus("Camera stopped");
  }

  function confirmPrediction() {
    if (!prediction?.prediction) return;
    setPatientMessage((old) =>
      [old.trim(), prediction.prediction].filter(Boolean).join(" ")
    );
    setPrediction(null);
  }

  async function sendPatientMessage() {
    const text = patientMessage.trim();
    if (!text) return;

    setBusy(true);
    try {
      const response = await fetch(`${API}/message/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Request failed");

      setHistory((old) => [
        ...old,
        { role: "Patient", text: result.original_text },
      ]);
      setPatientMessage("");
    } catch (error) {
      alert(`Could not send message: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function improveWithAI(text, setter) {
    if (!text.trim()) {
      setAiNote("Enter a message first.");
      return;
    }

    setAiBusy(true);
    setAiNote("");

    try {
      const response = await fetch(`${API}/genai/improve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), language }),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "GenAI request failed");
      }

      setter(result.suggestion);
      setAiNote("AI suggestion loaded. Review and edit it before using it.");
    } catch (error) {
      setAiNote(error.message || "GenAI is unavailable.");
    } finally {
      setAiBusy(false);
    }
  }

  function startListening() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setAiNote("Speech recognition is not supported in this browser.");
      return;
    }

    recognitionRef.current?.stop();

    const recognition = new SpeechRecognition();
    recognition.lang = languageSettings[language].speech;
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const spoken = event.results[0][0].transcript;
      setDoctorReply((old) => [old.trim(), spoken].filter(Boolean).join(" "));
    };

    recognition.onerror = (event) => {
      setAiNote(`Microphone error: ${event.error}`);
      setListening(false);
    };

    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    setAiNote("Listening... speak clearly into the microphone.");
    setListening(true);

    try {
      recognition.start();
    } catch {
      setListening(false);
      setAiNote("Could not start the microphone. Try again.");
    }
  }

  function readReplyAloud() {
    if (!doctorReply.trim() || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(doctorReply);
    utterance.lang = languageSettings[language].voice;
    window.speechSynthesis.speak(utterance);
  }

  function sendDoctorReply() {
    const text = doctorReply.trim();
    if (!text) return;

    setHistory((old) => [...old, { role: "Doctor", text, language }]);
    setDoctorReply("");
    setAiNote("");
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">✳</div>
          <div>
            <h1>SignBridge AI</h1>
            <p>Accessible healthcare communication</p>
          </div>
        </div>
        <div className="status">
          <span className={apiStatus === "Connected" ? "dot online" : "dot"} />
          API: {apiStatus}
          <button className="check-btn" onClick={checkAPI}>Check</button>
        </div>
      </header>

      <section className="welcome">
        <span className="eyebrow">PATIENT-FIRST CARE</span>
        <h2>Every sign deserves to be understood.</h2>
        <p>Connecting patients and healthcare professionals.</p>
      </section>

      <section className="dashboard">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-tag">01 / PATIENT</span>
              <h3>Patient communication</h3>
            </div>
            <span className="avatar patient-avatar">P</span>
          </div>

          <div className="camera-area">
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              playsInline
              muted
            />
            {cameraStatus !== "Camera running" && (
              <div className="camera-placeholder">
                <div className="camera-symbol">◎</div>
                <h4>Live sign recognition</h4>
                <p>Start the webcam and perform one of your trained signs.</p>
              </div>
            )}
          </div>

          <div className="camera-controls">
            <button
              className="primary-btn"
              onClick={startCamera}
              disabled={cameraStatus === "Camera running"}
            >
              Start webcam
            </button>
            <button
              className="secondary-btn"
              onClick={stopCamera}
              disabled={cameraStatus !== "Camera running"}
            >
              Stop webcam
            </button>
          </div>
          <p className="helper">{cameraStatus}</p>

          {cameraError && <p className="feature-note error-note">{cameraError}</p>}

          {prediction && (
            <div className="prediction-card">
              <span className="panel-tag">MODEL PREDICTION</span>
              <h4>{prediction.prediction}</h4>
              <p>
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </p>
              <button className="primary-btn" onClick={confirmPrediction}>
                Confirm and add sign
              </button>
              <p className="helper">
                Verify that the prediction matches the sign before using it.
              </p>
            </div>
          )}

          <label htmlFor="patient-message">Patient message</label>
          <textarea
            id="patient-message"
            value={patientMessage}
            onChange={(e) => setPatientMessage(e.target.value)}
            placeholder="Type or confirm the patient's message..."
            rows={3}
          />

          <button
            className="secondary-btn"
            disabled={aiBusy || !patientMessage.trim()}
            onClick={() => improveWithAI(patientMessage, setPatientMessage)}
          >
            {aiBusy ? "Working..." : "Improve message with GenAI"}
          </button>

          <div className="quick-heading">QUICK MESSAGES</div>
          <div className="quick-messages">
            {quickMessages.map((message) => (
              <button
                className="quick-btn"
                key={message}
                onClick={() => setPatientMessage(message)}
              >
                + {message}
              </button>
            ))}
          </div>

          <button
            className="primary-btn"
            disabled={busy || !patientMessage.trim()}
            onClick={sendPatientMessage}
          >
            {busy ? "Sending..." : "Send patient message →"}
          </button>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-tag">02 / DOCTOR</span>
              <h3>Healthcare professional</h3>
            </div>
            <span className="avatar doctor-avatar">D</span>
          </div>

          <div className="doctor-note">
            <span className="note-icon">✚</span>
            <div>
              <strong>Care team workspace</strong>
              <p>Write or dictate an accessible response.</p>
            </div>
          </div>

          <label htmlFor="reply-language">Reply language</label>
          <select
            id="reply-language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option>English</option>
            <option>Hindi</option>
            <option>Kannada</option>
          </select>

          <label htmlFor="doctor-reply">Doctor's reply</label>
          <textarea
            id="doctor-reply"
            value={doctorReply}
            onChange={(e) => setDoctorReply(e.target.value)}
            placeholder="Type or dictate a response..."
            rows={5}
          />

          <div className="camera-controls">
            <button
              className="secondary-btn"
              onClick={startListening}
              disabled={listening}
            >
              {listening ? "Listening..." : "🎙 Speak reply"}
            </button>
            <button
              className="secondary-btn"
              onClick={readReplyAloud}
              disabled={!doctorReply.trim()}
            >
              🔊 Read aloud
            </button>
          </div>

          <button
            className="secondary-btn"
            onClick={() => improveWithAI(doctorReply, setDoctorReply)}
            disabled={aiBusy || !doctorReply.trim()}
          >
            {aiBusy ? "Working..." : "Improve reply with GenAI"}
          </button>

          {aiNote && <p className="feature-note">{aiNote}</p>}

          <button
            className="primary-btn"
            disabled={!doctorReply.trim()}
            onClick={sendDoctorReply}
          >
            Add doctor reply →
          </button>

          <p className="helper">
            AI suggestions require review. Speech and language support depend
            on your browser and installed voices.
          </p>
        </article>
      </section>

      <section className="history-panel">
        <div className="history-heading">
          <div>
            <span className="panel-tag">03 / CONVERSATION</span>
            <h3>Communication history</h3>
          </div>
          <span className="message-count">{history.length} messages</span>
        </div>

        {history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">⌁</div>
            <strong>No messages yet</strong>
            <p>Confirmed messages and replies will appear here.</p>
          </div>
        ) : (
          <div className="message-list">
            {history.map((item, index) => (
              <div className="message-row" key={`${index}-${item.role}`}>
                <span className={`avatar ${
                  item.role === "Patient" ? "patient-avatar" : "doctor-avatar"
                }`}>
                  {item.role === "Patient" ? "P" : "D"}
                </span>
                <div className="message-content">
                  <div className="message-meta">
                    <strong>{item.role}</strong>
                    {item.language && <span>{item.language}</span>}
                  </div>
                  <p>{item.text}</p>
                  {item.role === "Doctor" && (
                    <button
                      className="quick-btn"
                      onClick={() => {
                        window.speechSynthesis.cancel();
                        const utterance = new SpeechSynthesisUtterance(item.text);
                        utterance.lang = languageSettings[item.language || "English"].voice;
                        window.speechSynthesis.speak(utterance);
                      }}
                    >
                      🔊 Play reply
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <footer>
        <span>SignBridge AI · Accessible communication for better care</span>
        <span>Prototype · Not for emergency reliance</span>
      </footer>
    </main>
  );
}

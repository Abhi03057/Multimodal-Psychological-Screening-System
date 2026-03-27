import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import cv2
import requests
import time
import os
import threading
from deepface import DeepFace

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
CHUNK_DURATION = 3
SAMPLE_RATE    = 22050
CHUNK_FILE     = "live_chunk.wav"
API_URL        = "http://127.0.0.1:8000/analysis/"
CAMERA_INDEX   = 1       # DroidCam (change to 0 for built-in webcam)
MIC_DEVICE     = 3       # Realtek Microphone Array

# ══════════════════════════════════════════════════════
#  SHARED STATE (updated by audio thread)
# ══════════════════════════════════════════════════════
audio_emotion  = "waiting..."
audio_energy   = "..."
face_emotion   = "waiting..."
face_confidence = 0.0
chunk_count    = 0
running        = True

# ══════════════════════════════════════════════════════
#  AUDIO RECORDING + API CALL (runs in background thread)
# ══════════════════════════════════════════════════════
def audio_loop():
    global audio_emotion, audio_energy, chunk_count, running
    while running:
        chunk_count += 1
        # Record audio
        recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32", device=MIC_DEVICE)
        sd.wait()
        write(CHUNK_FILE, SAMPLE_RATE, (recording * 32767).astype(np.int16))

        # Send to API
        try:
            with open(CHUNK_FILE, "rb") as f:
                res = requests.post(API_URL,
                                    files={"audio_file": (CHUNK_FILE, f, "audio/wav")},
                                    data={"phq9_score": 8, "gad7_score": 6})
            if res.status_code == 200:
                r = res.json()
                audio_emotion = r.get("detected_emotion", "unknown")
                audio_energy  = r.get("audio_features", {}).get("energy", "?")
        except Exception as e:
            audio_emotion = f"error: {e}"

        time.sleep(0.3)

    # Clean up
    if os.path.exists(CHUNK_FILE):
        os.remove(CHUNK_FILE)

# ══════════════════════════════════════════════════════
#  MAIN — CAMERA WINDOW + FACE ANALYSIS
# ══════════════════════════════════════════════════════
print("═" * 60)
print("🧠 MULTIMODAL LIVE ANALYSIS — with Camera Preview")
print(f"  🎤 Mic: Device {MIC_DEVICE} | 📷 Camera: Index {CAMERA_INDEX}")
print("═" * 60)
print("Press 'Q' in the camera window to stop.\n")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Could not open camera! Check DroidCam connection.")
    exit(1)

# Start audio thread
audio_thread = threading.Thread(target=audio_loop, daemon=True)
audio_thread.start()

frame_count = 0
analyze_every = 15  # Run DeepFace every N frames to keep it smooth

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run face emotion detection periodically (not every frame — too slow)
    if frame_count % analyze_every == 0:
        try:
            results = DeepFace.analyze(frame, actions=["emotion"],
                                       enforce_detection=False, silent=True)
            if isinstance(results, list):
                results = results[0]
            face_emotion = results["dominant_emotion"]
            face_confidence = results["emotion"][face_emotion] / 100.0
        except Exception:
            face_emotion = "unknown"
            face_confidence = 0.0

    # ── Draw overlay on the frame ──
    h, w = frame.shape[:2]

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "MULTIMODAL SCREENING", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Face emotion
    face_color = (0, 255, 0) if face_emotion in ["happy", "neutral", "calm"] else (0, 165, 255)
    cv2.putText(frame, f"Face: {face_emotion.upper()} ({face_confidence:.0%})",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    # Audio emotion
    audio_color = (0, 255, 0) if audio_emotion in ["happy", "neutral", "calm"] else (0, 165, 255)
    cv2.putText(frame, f"Audio: {audio_emotion.upper()} | Energy: {audio_energy}",
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)

    # Congruence indicator at bottom
    if face_emotion == audio_emotion:
        cv2.putText(frame, "CONGRUENT", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "INCONGRUENT", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show
    cv2.imshow("Multimodal Psychological Screening", frame)

    # Quit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
print("\n🛑 Session ended.")
